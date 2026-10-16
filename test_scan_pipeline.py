"""不接硬件：透视站间三角化、刚体鲁棒性、零偏保留与故障显式失败。"""
import unittest
import tempfile
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scan_pipeline as p
from offline_dic import SubsetMatcher


class PipelineTests(unittest.TestCase):
    def test_repeat_resume_captures_last_station_and_returns(self):
        import resume_reference_scan as resume
        import scan_capture as sc
        from test_scan_validation import SimRig, GEO
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'def';d.mkdir()
            r=Path(tmp)/'ref';r.mkdir()
            calib=d/'stereo_calib.npz'
            np.savez(calib,image_size=np.array([40,30]))
            (r/'meta.json').write_text(json.dumps(dict(status='completed',stations=[0,3000],calib=str(calib))))
            a=np.full((30,40),10,np.float32)
            sc.save_pair(str(d),0,a,a)
            sc.save_pair(str(d),1,a,a,prefix='candidate')
            original=(d/'station_00_L.png').read_bytes()
            meta=dict(role='def',status='aborted',stations=[0],planned_stations=[0,3000],
                      ref_dir=str(r),position_uncertain=False,pending_target_pulses=None,
                      candidate_station=1,origin_pulses=0,last_known_position_pulses=3000,
                      visited_range_pulses=[0,3000],start_encoder_deg=1000,
                      options={},avg=4,exposure_us=[50000,50000],calib=str(calib),
                      measured_shift_px=[[0,0]],station_records=[{}])
            (d/'meta.json').write_text(json.dumps(meta))
            def factory(**kwargs):
                rig=SimRig(**kwargs)
                rig.start_deg=rig.motor.deg=1337.5
                return rig
            with patch.object(resume,'Rig',side_effect=factory), \
                 patch.object(resume,'shift_px',return_value=(0,0,.99)), \
                 patch.object(sc,'shift_px',return_value=(0,0,.99)), \
                 patch.object(sc,'geometry_report',return_value=GEO), \
                 patch.object(sc,'texture_frac',return_value=1), \
                 patch.object(sc,'ROOT',tmp),patch.object(sc.time,'sleep'), \
                 contextlib.redirect_stdout(io.StringIO()):
                resume.main([str(d)])
            result=json.loads((d/'meta.json').read_text(encoding='utf-8'))
            self.assertEqual(result['stations'],[0,3000])
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['returned_to_origin'])
            self.assertEqual((d/'station_00_L.png').read_bytes(),original)
            self.assertTrue(list((d/'resume_history').glob('*/candidate_01_L.png')))

    def test_repeat_resume_requires_saved_exact_plan(self):
        from resume_reference_scan import validate_resume
        m=dict(role='def',status='aborted',stations=[0,1000],planned_stations=[0,1000,2000],
               position_uncertain=False,pending_target_pulses=None,candidate_station=2,
               origin_pulses=-300,last_known_position_pulses=1700)
        self.assertEqual(validate_resume(m),2)
        for delta in (dict(planned_stations=[0,1000]),dict(planned_stations=[0,999,2000]),
                      dict(last_known_position_pulses=2000),dict(position_uncertain=True)):
            with self.assertRaises(ValueError):
                validate_resume(dict(m,**delta))

    def test_robust_rigid_planar_outliers(self):
        rng=np.random.default_rng(3)
        A=np.column_stack([rng.uniform(-50,50,(300,2)),np.full(300,500.)])
        R0=Rotation.from_euler('xyz',[1,3,2],degrees=True).as_matrix()
        t0=np.array([55.,0.2,12.])
        B=A@R0.T+t0+rng.normal(0,.01,A.shape)
        B[:50]+=rng.normal(0,5,(50,3))
        R,t,ok,res=p.robust_rigid(A,B)
        self.assertGreater(ok.sum(),245)
        self.assertLess(ok[:50].sum(),3)
        np.testing.assert_allclose(R,R0,atol=.0002)
        np.testing.assert_allclose(t,t0,atol=.08)
        self.assertLess(np.percentile(res[ok],95),.04)

    def test_bad_links_fail(self):
        with self.assertRaises(RuntimeError):
            p.robust_rigid(np.zeros((29,3)),np.zeros((29,3)))
        with self.assertRaises(RuntimeError):
            p.robust_rigid(np.ones((100,3)),np.ones((100,3)))
        rng=np.random.default_rng(4)
        with self.assertRaises(RuntimeError):
            p.robust_rigid(rng.normal(size=(100,3)),rng.normal(size=(100,3)))

    def test_overlap_preserves_anchor_and_global_offset(self):
        x,y=np.meshgrid(np.arange(20),np.arange(5))
        P=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        global_bias=np.array([.2,-.01,.04])
        D=[np.tile(global_bias+d,(len(P),1)) for d in
           ([0,0,0],[.03,.02,-.06],[-.1,.04,.02])]
        offsets,seams=p.overlap_correction([P,P,P],D)
        np.testing.assert_array_equal(offsets[0],np.zeros(3))
        for d,c in zip(D,offsets):
            np.testing.assert_allclose(d+c,np.tile(global_bias,(len(P),1)),atol=1e-12)
        self.assertLess(seams[-1]['corrected_p95_norm_um'],1e-9)
        with self.assertRaises(RuntimeError):
            p.overlap_correction([P,P+100],[D[0],D[1]])

    def test_projective_link_uses_exact_point_not_grid(self):
        class Stereo:
            out_size=(1024,768)
            epi_thresh=1.5
            def _epi_err(self,x,y,xr,yr):
                return abs(y-yr)
            def _triangulate(self,x,y,xr,yr):
                z=100000/(x-xr)
                return np.column_stack([x*z/1000,y*z/1000,z])
        eng=Stereo()
        rng=np.random.default_rng(28)
        a=cv2.GaussianBlur(rng.uniform(0,255,(768,1024)).astype(np.float32),(0,0),1)
        Hba=np.float64([[.96,0,40],[0,.96,6],[0,0,1]])
        b=cv2.warpPerspective(a,np.linalg.inv(Hba),eng.out_size)
        def view(im,z):
            return dict(lp=im,rw=im,Hi=np.float64([[1,0,-100000/z],[0,1,0],[0,0,1]]),
                        texture=np.ones(im.shape,bool))
        va,vb=view(a,500),view(b,480)
        xs,ys=np.meshgrid(np.arange(260,950,40),np.arange(80,690,40))
        x,y=xs.ravel(),ys.ravel()
        st=dict(px=x,py=y,P=np.column_stack([x*.5,y*.5,np.full(len(x),500)]))
        mat=SubsetMatcher(half=16,search=5,min_ncc=.7)
        R,t,rep,data=p.link_stations(eng,mat,va,vb,st)
        np.testing.assert_allclose(R,np.eye(3),atol=.0003)
        np.testing.assert_allclose(t,[20,3,20],atol=.15)
        self.assertGreater(rep['inliers'],200)
        self.assertLess(rep['p95_mm'],.12)
        xy=np.array(data['pixel_pairs'])
        self.assertGreater(np.mean(np.abs(xy[:,2]-np.rint(xy[:,2]))),.1)
        # 同图时序必须为零，不能由重叠校正制造零。
        same=p.process_station(eng,mat,va,va,32)
        np.testing.assert_allclose(same['D'],0,atol=1e-6)


if __name__=='__main__':
    unittest.main(verbosity=2)
