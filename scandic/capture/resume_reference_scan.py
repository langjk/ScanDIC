"""从几何闸门中止的候选站续接ref/def扫描；不支持位置不明或电机故障后恢复。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from scandic.capture.scan_capture import (Rig, ScanLog, motion, capture_station, save_pair,
                          texture_frac, shift_px, require_shift)


def validate_resume(meta):
    k = len(meta.get('stations', []))
    if (meta.get('role') not in ('ref', 'def') or meta.get('status') != 'aborted' or k < 1 or
            meta.get('position_uncertain', True) or meta.get('pending_target_pulses') is not None or
            meta.get('candidate_station') != k):
        raise ValueError('Only a known, stopped candidate position can resume')
    if meta['role'] == 'ref':
        step = meta.get('step_pulses', 0)
        if (not meta.get('options', {}).get('auto_edge') or step <= 0 or
                meta['stations'] != [j*step for j in range(k)]):
            raise ValueError('Invalid station plan')
        offset = k*step
    else:
        plan = meta.get('planned_stations', [])
        if (len(plan) <= k or plan[0] != 0 or plan[:k] != meta['stations'] or
                any(b <= a for a,b in zip(plan, plan[1:]))):
            raise ValueError('Invalid repeat station plan')
        offset = plan[k]
    if meta['last_known_position_pulses'] != meta['origin_pulses']+offset:
        raise ValueError('Last position does not match candidate station')
    return k


def read_pair(d, prefix, k):
    result = []
    for tag in ('L', 'R'):
        path = d/f'{prefix}_{k:02d}_{tag}.png'
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(str(path))
        result.append(a.astype(np.float32)/(256 if a.dtype == np.uint16 else 1))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    args = ap.parse_args(argv)
    d = Path(args.directory).resolve()
    meta = json.loads((d/'meta.json').read_text(encoding='utf-8'))
    k = validate_resume(meta)
    if meta['role'] == 'def':
        reference = json.loads((Path(meta['ref_dir'])/'meta.json').read_text(encoding='utf-8'))
        if (reference['status'] != 'completed' or reference['stations'] != meta['planned_stations'] or
                Path(reference['calib']).read_bytes() != Path(meta['calib']).read_bytes()):
            raise ValueError('Repeat plan/reference calibration mismatch')
    candidate = read_pair(d, 'candidate', k)
    first = read_pair(d, 'station', 0)[0]
    prev = read_pair(d, 'station', k-1)[0]
    with np.load(meta['calib']) as c:
        if tuple(c['image_size']) != first.shape[::-1]:
            raise ValueError('Calibration/image size mismatch')
    log = None
    rig = None
    try:
        rig = Rig(budget=meta.get('budget_pulses'))
        expected_deg = meta['start_encoder_deg']+meta['last_known_position_pulses']*360/3200
        if abs(rig.start_deg-expected_deg) > 3.6:
            raise RuntimeError('Encoder changed since saved stop; no motion allowed')
        if not np.allclose(rig.exposures, meta['exposure_us'], atol=1, rtol=0):
            raise RuntimeError('Exposure changed; no motion allowed')
        current = rig.pair_avg(1)
        for old, new in zip(candidate, current):
            dx, dy, ncc = shift_px(old, new)
            require_shift(dx, dy, ncc, minimum=0.8)
            if np.hypot(dx, dy) >= 2:
                raise RuntimeError('Image moved from candidate position; no motion allowed')
        history = d/'resume_history'/time.strftime('%Y%m%d_%H%M%S')
        history.mkdir(parents=True)
        shutil.copy2(d/'meta.json', history/'meta.json')
        for tag in ('L', 'R'):
            shutil.copy2(d/f'candidate_{k:02d}_{tag}.png', history/f'candidate_{k:02d}_{tag}.png')
        rig.start_deg = meta['start_encoder_deg']
        rig.pos = meta['last_known_position_pulses']
        rig.origin = meta['origin_pulses']
        rig.min_pos, rig.max_pos = meta['visited_range_pulses']
        meta.setdefault('resume_history', []).append(str(history))
        meta.pop('error', None)
        log = ScanLog(str(d), meta, resume=True)
        log.event(f'从候选站{k}续扫，编码器和双目图像已确认；保留既有站位',
                  status='running', geometry_policy='SIFT warning confirmed by unconstrained subset ECC')
        while True:
            if k != len(meta['stations']):
                raise RuntimeError('Station index mismatch')
            if meta['role'] == 'def' and k == len(meta['planned_stations']):
                log.event('基准站位已全部复现', termination_reason='plan_completed')
                break
            limit = meta['options'].get('stations')
            if limit is not None and k >= limit:
                raise RuntimeError('Station limit reached before board edge')
            offset = meta['planned_stations'][k] if meta['role'] == 'def' else k*meta['step_pulses']
            motion(rig, log, rig.origin+offset, f'续扫站位{k}')
            if meta['role'] == 'ref':
                a, b = rig.pair_avg(1)
                tf = texture_frac(a)
                if tf < meta['options']['edge_frac']:
                    save_pair(str(d), -2, a, b)
                    log.event(f'另一侧板边触发，覆盖{tf:.1%}', termination_reason='board_edge',
                              end_edge_position_pulses=rig.pos, end_edge_texture_fraction=tf)
                    break
            prev, _ = capture_station(rig, log, offset, meta['avg'], meta['calib'], prev)
            k += 1
        motion(rig, log, rig.origin, '正常回原点')
        cur, _ = rig.pair_avg(1)
        dx, dy, ncc = shift_px(first, cur)
        require_shift(dx, dy, ncc)
        if np.hypot(dx, dy) >= 30:
            raise RuntimeError('Return image error exceeds 30px')
        log.event(f'续扫完成并回原点，图像残差({dx:+.2f},{dy:+.2f})px NCC={ncc:.3f}',
                  status='completed', returned_to_origin=True, candidate_station=None,
                  return_check={'shift_px': [dx, dy], 'ncc': ncc,
                                'encoder_deg': rig.motor.read_position_deg(), 'motor_status': rig.check_status()})
    except BaseException as exc:
        if rig:
            rig.stop()
        if log:
            log.event(f'续扫中止，不自动回程: {exc}', status='aborted', error=repr(exc),
                      position_uncertain=bool(rig and rig.faulted))
        raise
    finally:
        if rig:
            rig.close(home=False)


if __name__ == '__main__':
    main()
