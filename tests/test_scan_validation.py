"""不加载相机 SDK、不打开串口的匹配/扫描回归验证。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import fourier_shift

import scandic.capture.scan_capture as sc
from scandic.dic_engine import contrast_normalize
from scandic.processing.offline_dic import SubsetMatcher, load_state
from scandic.hardware.zdt_motor import ZdtMotor


OK = dict(enabled=True, in_position=True, stalled=False, stall_protect=False,
          limit_left=False, limit_right=False)
GEO = dict(sift_inliers=100, sift_matches=120, epipolar_median_full_px=0.3,
           epipolar_p95_full_px=0.6)


class Motor:
    def __init__(self):
        self.deg = 1000.0
        self.commands = []
        self.stopped = False
        self.closed = False
        self.fail = False

    def read_status(self):
        return dict(OK)

    def read_position_deg(self):
        return self.deg

    def move_pulses(self, p, **kw):
        self.commands.append((p, kw))

    def wait_arrival(self, **kw):
        if self.fail:
            raise RuntimeError('injected stall')
        self.deg += self.commands[-1][0] * 360 / 3200
        if abs(self.deg - kw['target_deg']) > 0.06:
            raise AssertionError('wrong encoder target')

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class SimRig(sc.Rig):
    instances = []

    def __init__(self, budget=None, exposure_us=None):
        self.budget = budget
        self.pos = self.origin = self.min_pos = self.max_pos = 0
        self.faulted = False
        self.camL = self.camR = None
        self.motor = Motor()
        self.test_motor = self.motor
        self.start_deg = self.motor.deg
        self.exposures = [50000, 50000]
        self.instances.append(self)

    def pair_avg(self, n):
        # 数组首像素编码当前位置，用于独立几何模拟。
        a = np.full((30, 40), self.pos, np.float32)
        return a, a.copy()


def fake_shift(a, b):
    return float(b[0, 0] - a[0, 0]) * 0.005, 0.0, 0.99


def fake_texture(a):
    return 0.3 if a[0, 0] < -3500 or a[0, 0] > 5000 else 1.0


class MatchingTests(unittest.TestCase):
    def test_known_shifts_and_input_preservation(self):
        rng = np.random.default_rng(18)
        a = cv2.GaussianBlur(rng.normal(128, 40, (384, 384)).astype(np.float32), (0, 0), 1.2)
        matcher = SubsetMatcher(half=32, search=10)
        errors = []
        for dx, dy in [(0, 0), (3.25, 2.25), (-3.25, -2.25), (0.5, -0.5), (5, -4)]:
            b = np.fft.ifftn(fourier_shift(np.fft.fftn(a), (dy, dx))).real.astype(np.float32)
            a_before, b_before = a.copy(), b.copy()
            estimates = []
            for y in (96, 192, 288):
                for x in (96, 192, 288):
                    result = matcher.match(a, b, x, y)
                    self.assertIsNotNone(result)
                    estimates.append(result[:2])
            err = np.asarray(estimates) - [dx, dy]
            errors.extend(np.abs(err).ravel())
            print('SHIFT', (dx, dy), 'median', np.median(estimates, axis=0).tolist())
            np.testing.assert_array_equal(a, a_before)
            np.testing.assert_array_equal(b, b_before)
        # 平滑纹理上的相位相关质心估计并非无偏；该测试验证符号回归，
        # 不把一个合成例子的0.01px误差当作普适精度承诺。
        self.assertLess(np.percentile(errors, 95), 0.3)

    def test_even_input_with_odd_optimal_dft(self):
        rng = np.random.default_rng(9)
        a = rng.normal(size=(122, 98)).astype(np.float32)
        from scandic.processing.stitch import phase_correlate_even
        shift, _ = phase_correlate_even(a, a, cv2.createHanningWindow((98, 122), cv2.CV_32F))
        np.testing.assert_allclose(shift, [0, 0], atol=0.001)

    def test_float_png_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 32
            sc.save_pair(tmp, 0, a, a)
            Path(tmp, 'station_00_L.png').rename(Path(tmp, 'state_0000_L.png'))
            decoded = load_state(tmp, 0, 'L')
            self.assertEqual(decoded.dtype, np.float32)
            np.testing.assert_array_equal(decoded, a)
            self.assertEqual(contrast_normalize(decoded).dtype, np.float32)
            self.assertGreater(np.count_nonzero(decoded % 1), 0)

    def test_registration_direction(self):
        rng = np.random.default_rng(8)
        world = cv2.GaussianBlur(rng.normal(128, 40, (620, 900)).astype(np.float32), (0, 0), 2)
        a = world[50:562, 100:868].copy()
        b = world[74:586, 20:788].copy()
        dx, dy, ncc = sc.shift_px(a, b)
        np.testing.assert_allclose([dx, dy], [-80, 24], atol=0.2)
        self.assertGreater(ncc, 0.95)

    def test_projective_registration_and_dark_texture(self):
        rng = np.random.default_rng(25)
        world = cv2.GaussianBlur(rng.uniform(0, 255, (768, 1024)).astype(np.float32), (0, 0), 1)
        Hba = np.float64([[0.96, 0, 130], [0, 0.98, 10], [0, 0, 1]])
        b = cv2.warpPerspective(world, np.linalg.inv(Hba), (1024, 768))
        estimated, ncc, count = sc.projective_registration(world, b)
        points = np.float32([[[300, 300]], [[600, 500]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(points, estimated),
                                   cv2.perspectiveTransform(points, Hba), atol=1)
        self.assertGreater(ncc, 0.8)
        self.assertGreaterEqual(count, 30)
        # 实心噪声散斑本身对比度较低：用几何点纹理检验照明不变性。
        speckle = np.zeros((768, 1024), np.float32)
        for x, y in rng.integers([0, 0], [1024, 768], (12000, 2)):
            cv2.circle(speckle, (int(x), int(y)), 2, 200, -1)
        self.assertGreater(sc.texture_frac(speckle * 0.25), 0.95)
        self.assertEqual(sc.texture_frac(np.full((768, 1024), 20, np.float32)), 0)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.sleep_patch = patch.object(sc.time, 'sleep', return_value=None)
        self.sleep_patch.start()

    def tearDown(self):
        self.sleep_patch.stop()

    def test_budget_before_command_and_mode2(self):
        rig = SimRig(budget=40000)
        rig.goto_abs(-20000)
        rig.goto_abs(20000)
        self.assertTrue(all(kw['mode'] == 2 for _, kw in rig.motor.commands))
        with self.assertRaises(RuntimeError):
            rig.goto_abs(20001)
        self.assertEqual(len(rig.motor.commands), 2)

    def test_stall_stops_and_forbids_home(self):
        rig = SimRig()
        motor = rig.motor
        motor.fail = True
        with self.assertRaises(RuntimeError):
            rig.goto_abs(100)
        self.assertTrue(rig.faulted and motor.stopped)
        with self.assertRaises(RuntimeError):
            rig.close(home=True)
        self.assertEqual(len(motor.commands), 1)
        self.assertTrue(motor.closed)

    def test_encoder_drift_prevents_motion(self):
        rig = SimRig()
        rig.motor.deg += 10
        with self.assertRaises(RuntimeError):
            rig.goto_abs(100)
        self.assertEqual(rig.motor.commands, [])

    def test_bad_arguments_before_hardware(self):
        for argv in [('ref', '--budget', '0'), ('def',),
                     ('ref', '--overlap', 'nan'), ('ref', '--stations', '0')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sc.parse_args(argv)

    def test_unlimited_total_uses_bounded_segments(self):
        rig = SimRig()
        log = type('Log', (), {'event': lambda *args, **kw: None})()
        self.assertIsNone(sc.parse_args(['ref']).budget)
        sc.motion(rig, log, 80001, 'long travel')
        self.assertEqual(rig.pos, 80001)
        self.assertLessEqual(max(abs(p) for p, _ in rig.motor.commands), sc.MAX_SEGMENT_PULSES)

    def test_segment_fault_stops_remaining_travel(self):
        rig = SimRig()
        rig.motor.fail = True
        log = type('Log', (), {'event': lambda *args, **kw: None})()
        with self.assertRaises(RuntimeError):
            sc.motion(rig, log, 80001, 'long travel')
        self.assertEqual(len(rig.motor.commands), 1)
        self.assertTrue(rig.motor.stopped)

    def test_nonfinite_registration_rejected(self):
        for vals in [(0, 0, float('nan')), (float('nan'), 0, 1), (0, 0, 0.1)]:
            with self.assertRaises(RuntimeError):
                sc.require_shift(*vals)

    def test_alignment_must_converge(self):
        rig = SimRig()
        log = type('Log', (), {'event': lambda *args, **kw: None})()
        with patch.object(sc, 'shift_px', return_value=(100, 0, 0.9)):
            with self.assertRaisesRegex(RuntimeError, '未收敛'):
                sc.align_origin(rig, log, np.zeros((2, 2)), (1, 0))
        self.assertEqual(len(rig.motor.commands), 4)

    def test_wait_arrival_checks_stale_flag_and_fault(self):
        motor = object.__new__(ZdtMotor)
        motor.read_status = lambda: dict(OK)
        readings = iter([0, 100])
        motor.read_position_deg = lambda: next(readings)
        motor.stop = lambda: None
        self.assertTrue(motor.wait_arrival(target_deg=100))
        motor.read_status = lambda: dict(OK, limit_left=True)
        with self.assertRaises(RuntimeError):
            motor.wait_arrival(target_deg=100)

    def run_simulation(self, tmp, *extra):
        calib = Path(tmp, 'input.npz')
        np.savez(calib, **{k: np.ones(3) for k in ('K1', 'K2', 'D1', 'D2', 'R', 'T', 'image_size')})
        with patch.object(sc, 'ROOT', tmp), patch.object(sc, 'Rig', SimRig), \
                patch.object(sc, 'geometry_report', return_value=GEO), \
                patch.object(sc, 'texture_frac', side_effect=fake_texture), \
                patch.object(sc, 'shift_px', side_effect=fake_shift), \
                contextlib.redirect_stdout(io.StringIO()):
            return sc.main(['ref', '--seek-edge', '--auto-edge', '--calib', str(calib), *extra])

    def test_full_reference_and_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.run_simulation(tmp)
            meta = json.loads(Path(out, 'meta.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'completed')
            self.assertEqual(meta['termination_reason'], 'board_edge')
            self.assertTrue(meta['returned_to_origin'])
            self.assertEqual(len(meta['stations']), 3)
            self.assertEqual(meta['stations'][0], 0)
            self.assertTrue(Path(out, 'station_02_R.png').exists())
            self.assertTrue(Path(tmp, 'WORK_LOG.md').exists())
            sc.validate_plan(meta)

    def test_seek_budget_abort_has_no_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, '预算'):
                self.run_simulation(tmp, '--budget', '4000')
            meta_file = next(Path(tmp, 'captures').glob('*/meta.json'))
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'aborted')
            self.assertFalse(meta['returned_to_origin'])
            self.assertEqual(meta['stations'], [])
            rig = SimRig.instances[-1]
            self.assertEqual(rig.pos, 0)  # 首次反向步进会超过探测时已访问的跨度
            self.assertTrue(rig.test_motor.stopped)
            self.assertEqual(len(rig.test_motor.commands), 2)

    def test_station_limit_does_not_claim_board_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, '站数上限'):
                self.run_simulation(tmp, '--stations', '2')
            meta_file = next(Path(tmp, 'captures').glob('*/meta.json'))
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            self.assertEqual(meta['status'], 'aborted')
            self.assertFalse(meta['returned_to_origin'])
            self.assertEqual(len(meta['stations']), 2)
            with self.assertRaises(ValueError):
                sc.validate_plan(meta)

    def test_bad_geometry_prevents_probe(self):
        rig = SimRig()
        log = type('Log', (), {'event': lambda *args, **kw: None})()
        args = sc.parse_args(['ref', '--seek-edge'])
        with patch.object(sc, 'geometry_report', return_value=dict(GEO, epipolar_median_full_px=10)):
            with self.assertRaisesRegex(RuntimeError, '标定不符'):
                sc.run_reference(args, rig, log, 'unused')
        self.assertEqual(rig.motor.commands, [])

    def test_geometry_subset_confirmation_keeps_threshold(self):
        v = dict(ecc_count=355, ecc_median_abs_px=0.741, ecc_p95_abs_px=1.044)
        report = dict(GEO, epipolar_median_full_px=0.862, subset_verification=v)
        sc.require_geometry(report)
        for bad in [dict(v, ecc_count=29), dict(v, ecc_median_abs_px=0.801),
                    dict(v, ecc_p95_abs_px=1.501), dict(v, ecc_median_abs_px=float('nan'))]:
            with self.assertRaises(RuntimeError):
                sc.require_geometry(dict(report, subset_verification=bad))

    def test_resume_requires_known_candidate_position(self):
        from scandic.capture.resume_reference_scan import validate_resume
        meta = dict(role='ref', status='aborted', stations=[0, 1000], step_pulses=1000,
                    origin_pulses=-3000, last_known_position_pulses=-1000,
                    position_uncertain=False, pending_target_pulses=None,
                    candidate_station=2, options={'auto_edge': True})
        self.assertEqual(validate_resume(meta), 2)
        for changes in [dict(position_uncertain=True), dict(pending_target_pulses=1000),
                        dict(last_known_position_pulses=0), dict(candidate_station=1),
                        dict(status='completed'), dict(stations=[0, 999])]:
            with self.assertRaises(ValueError):
                validate_resume(dict(meta, **changes))

    def test_resume_completes_without_overwriting_accepted_pairs(self):
        import scandic.capture.resume_reference_scan as resume
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp, 'scan'); d.mkdir()
            calib = d/'stereo_calib.npz'
            np.savez(calib, image_size=np.array([40, 30]))
            a = np.full((30, 40), 10, np.float32)
            for k in (0, 1):
                sc.save_pair(str(d), k, a, a)
            sc.save_pair(str(d), 2, a, a, prefix='candidate')
            first_bytes = (d/'station_00_L.png').read_bytes()
            meta = dict(role='ref', status='aborted', position_uncertain=False,
                        pending_target_pulses=None, stations=[0, 3000], step_pulses=3000,
                        origin_pulses=0, last_known_position_pulses=6000,
                        visited_range_pulses=[0, 6000], start_encoder_deg=1000,
                        candidate_station=2, options={'auto_edge': True, 'edge_frac': 0.55},
                        avg=4, exposure_us=[50000, 50000], calib=str(calib),
                        measured_shift_px=[[0, 0], [0, 0]], station_records=[{}, {}])
            (d/'meta.json').write_text(json.dumps(meta), encoding='utf-8')
            def factory(**kwargs):
                rig = SimRig(**kwargs)
                rig.motor.deg = rig.start_deg = 1675.0
                return rig
            with patch.object(resume, 'Rig', side_effect=factory), \
                    patch.object(resume, 'shift_px', return_value=(0, 0, 0.99)), \
                    patch.object(sc, 'shift_px', return_value=(0, 0, 0.99)), \
                    patch.object(resume, 'texture_frac', side_effect=lambda a: 0.2 if a[0, 0] >= 12000 else 1), \
                    patch.object(sc, 'texture_frac', return_value=1), \
                    patch.object(sc, 'geometry_report', return_value=GEO), \
                    patch.object(sc, 'ROOT', tmp), contextlib.redirect_stdout(io.StringIO()):
                resume.main([str(d)])
            result = json.loads((d/'meta.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['stations'], [0, 3000, 6000, 9000])
            self.assertTrue(result['returned_to_origin'])
            self.assertEqual(first_bytes, (d/'station_00_L.png').read_bytes())
            self.assertTrue(list((d/'resume_history').glob('*/candidate_02_L.png')))

    def test_metadata_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'scan')
            log = sc.ScanLog(str(out), {'status': 'running'})
            original = sc.os.replace
            calls = []
            def busy_then_replace(a, b):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('transient sharing violation')
                return original(a, b)
            with patch.object(sc, 'ROOT', tmp), patch.object(sc.os, 'replace', side_effect=busy_then_replace), \
                    contextlib.redirect_stdout(io.StringIO()):
                log.event('saved', status='completed')
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads((out/'meta.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
