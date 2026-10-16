"""静止实验不应产生额外运动，编码器或起始位置异常必须停止。"""
import contextlib
import io
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
import scan_capture as sc
import capture_static_station as static
from test_scan_validation import SimRig


class StaticTests(unittest.TestCase):
    def test_temporal_common_points_follow_pixel_identity(self):
        from process_static_station import common_cube
        x=np.arange(300)
        reverse=np.arange(299,-1,-1)
        states=[dict(px=x,py=np.zeros(300)),dict(px=x[reverse],py=np.zeros(300))]
        a=np.column_stack([x,x*2,x*3]).astype(float)
        pixels,cube=common_cube(states,[a,(a+1)[reverse]])
        np.testing.assert_array_equal(pixels[:,0],x)
        np.testing.assert_array_equal(cube[1]-cube[0],np.ones((300,3)))
        with self.assertRaises(RuntimeError):
            common_cube([dict(px=x[:100],py=np.zeros(100))],[a[:100]])

    def test_geometry_diagnostic_does_not_change_gate(self):
        from test_scan_validation import GEO
        self.assertTrue(static.geometry_passed(GEO))
        self.assertFalse(static.geometry_passed(dict(GEO,epipolar_median_full_px=1.1)))

    def test_series_has_no_movement_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig=SimRig()
            log=sc.ScanLog(tmp,dict(states=[]),resume=True)
            with patch.object(sc,'ROOT',tmp),contextlib.redirect_stdout(io.StringIO()):
                static.collect_static(rig,log,3,4,0)
            self.assertEqual(rig.motor.commands,[])
            self.assertEqual(len(log.meta['states']),3)

    def test_encoder_drift_stops_static_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig=SimRig()
            log=sc.ScanLog(tmp,dict(states=[]),resume=True)
            with patch.object(rig.motor,'read_position_deg',side_effect=[1000,1001]):
                with self.assertRaises(RuntimeError):
                    static.collect_static(rig,log,3,4,0)
            self.assertEqual(log.meta['states'],[])
            self.assertEqual(rig.motor.commands,[])

    def test_source_must_be_at_known_origin(self):
        m=dict(status='completed',returned_to_origin=True,position_uncertain=False,
               pending_target_pulses=None,last_known_position_pulses=0,origin_pulses=0,
               stations=[0,1000,2000])
        self.assertEqual(static.validate_source(m,2),2000)
        for changes in (dict(last_known_position_pulses=1000),dict(position_uncertain=True),
                        dict(returned_to_origin=False),dict(status='aborted')):
            with self.assertRaises(ValueError):
                static.validate_source(dict(m,**changes),2)


if __name__=='__main__':
    unittest.main()
