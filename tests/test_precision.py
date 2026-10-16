import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from scipy.ndimage import fourier_shift
from scandic.processing.precise_match import SplineImage,PreciseMatcher
from scandic.processing.stereo_temporal import track


class PrecisionTests(unittest.TestCase):
    def test_production_default_preserves_global_offset(self):
        import scandic.processing.precision_scan_pipeline as pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'scan';d.mkdir();cache=root/'cache';cache.mkdir()
            calib=root/'calib.npz';np.savez(calib,image_size=[100,100])
            x,y=np.meshgrid(np.arange(15),np.arange(15))
            P=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,500)])
            D=np.tile([.2,0,0],(len(P),1))
            for k in range(2):
                np.savez(cache/f'station_{k:02d}.npz',P=P,D=D,px=x.ravel(),py=y.ravel())
            dm=dict(calib=str(calib),stations=[0,1000])
            rep=dict(inliers=225,count=225,p95_mm=.01)
            with patch.object(pipeline,'validate_inputs',return_value=(root,{},dm,'hash')), \
                 patch.object(pipeline,'verified_cache'),patch.object(pipeline,'DicEngine'), \
                 patch.object(pipeline,'prepare',return_value={}),patch.object(pipeline,'log'), \
                 patch.object(pipeline,'link_stations',return_value=(np.eye(3),np.array([10.,0,0]),rep,{})):
                pipeline.main([str(d),'--cache',str(cache)])
            with np.load(d/'precision_scan_results/merged.npz') as z:
                np.testing.assert_array_equal(z['D'],z['D_raw'])
                np.testing.assert_allclose(z['D'][:,0],.2)
            summary=json.loads((d/'precision_scan_results/summary.json').read_text())
            self.assertFalse(summary['absolute_displacement_identifiable'])
            self.assertNotIn('overlap_diagnostic',summary)

    def test_global_specimen_and_camera_translation_are_ambiguous(self):
        from scandic.processing.camera_motion import project
        P=np.array([[-20.,-10,500],[20,10,500],[0,20,520]])
        K=np.diag([15000.,15000,1])
        R=cv2.Rodrigues(np.array([0,.4,0]))[0];t=np.array([-210.,0,30.])
        d=np.array([.2,.01,-.05])
        np.testing.assert_allclose(project(P+d,K),project(P,K,np.eye(3),d))
        np.testing.assert_allclose(project(P+d,K,R,t),project(P,K,R,t+R@d))

    def test_subpixel_origin_alignment_rejects_former_4px_offset(self):
        from unittest.mock import patch
        import scandic.capture.scan_capture as sc
        from tests.test_scan_validation import SimRig
        rig=SimRig();log=type('Log',(),{'event':lambda *args,**kwargs:None})()
        with patch.object(sc,'shift_px',side_effect=[(4.,0,.99),(.1,0,.99)]),patch.object(sc.time,'sleep'):
            sc.align_origin(rig,log,np.zeros((2,2)),(.04,0))
        self.assertEqual(len(rig.motor.commands),1)
        self.assertEqual(rig.motor.commands[0][0],-100)
        rig=SimRig()
        with patch.object(sc,'shift_px',return_value=(4.,.5,.99)):
            with self.assertRaisesRegex(RuntimeError,'横向'):
                sc.align_origin(rig,log,np.zeros((2,2)),(.04,0))
        self.assertEqual(rig.motor.commands,[])

    def test_camera_correction_requires_anchors_and_preserves_real_deformation(self):
        from scandic.processing.camera_motion import project,anchored_displacement
        rng=np.random.default_rng(72)
        xy=rng.uniform([-70,-45],[70,45],(500,2))
        P=np.column_stack([xy,np.full(500,550.)])
        K=np.array([[15000.,0,2048],[0,15000.,1500],[0,0,1]])
        R0=cv2.Rodrigues(np.array([0,.4,0]))[0];T0=np.array([-210.,0,35.])
        q0=np.column_stack([project(P,K),project(P,K,R0,T0)])
        anchors=xy[:,0]<-15
        # Deformation on the unanchored half must survive camera-motion removal.
        D=np.zeros_like(P);D[xy[:,0]>15]=[.025,-.01,.06]
        Rl=cv2.Rodrigues(np.array([.00004,-.00008,.00002]))[0];tl=np.array([.16,.02,-.03])
        Rdr=cv2.Rodrigues(np.array([-.00002,.00006,.00001]))[0];tdr=np.array([.1,-.01,.02])
        q1=np.column_stack([project(P+D,K,Rl,tl),project((P+D)@R0.T+T0,K,Rdr,tdr)])
        with self.assertRaises(ValueError):
            anchored_displacement(P,q0,q1,K,K,R0,T0,None)
        recovered,_=anchored_displacement(P,q0,q1,K,K,R0,T0,anchors)
        np.testing.assert_allclose(recovered,D,atol=2e-5)

    def test_origin_alignment_uses_only_stationary_roi(self):
        import scandic.capture.scan_capture as sc
        from tests.test_scan_validation import SimRig
        rig=SimRig();log=type('Log',(),{'event':lambda *args,**kwargs:None})()
        reference=np.arange(300*400).reshape(300,400)
        current=reference+7
        with patch.object(rig,'pair_avg',return_value=(current,current)) as acquire, \
             patch.object(sc,'shift_px',return_value=(.1,.1,.99)) as shift:
            sc.align_origin(rig,log,reference,(.04,0),roi=(100,50,128,150))
        acquire.assert_called_once_with(4)
        self.assertTrue(shift.call_args.kwargs['precise'])
        np.testing.assert_array_equal(shift.call_args.args[0],reference[50:200,100:228])
        np.testing.assert_array_equal(shift.call_args.args[1],current[50:200,100:228])
        self.assertEqual(rig.motor.commands,[])
        for roi in ((-1,0,128,128),(0,0,127,128),(300,0,128,128)):
            with patch.object(rig,'pair_avg') as acquire:
                with self.assertRaisesRegex(ValueError,'ROI'):
                    sc.align_origin(rig,log,reference,(.04,0),roi=roi)
                acquire.assert_not_called()

    def test_origin_estimator_resolves_subpixel_gate_and_sign(self):
        import scandic.capture.scan_capture as sc
        rng=np.random.default_rng(18)
        a=cv2.GaussianBlur(rng.normal(128,35,(384,384)).astype(np.float32),(0,0),1.3)
        for dx,dy in ((.15,.1),(.4,0),(-3.7,.1)):
            b=np.fft.ifftn(fourier_shift(np.fft.fftn(a),(dy,dx))).real.astype(np.float32)
            got=sc.shift_px(a,b,precise=True)
            np.testing.assert_allclose(got[:2],[-dx,-dy],atol=.008)
            self.assertEqual(np.hypot(*got[:2])<.3,np.hypot(dx,dy)<.3)

    def test_known_fractional_shift_photometry_and_fractional_query(self):
        rng=np.random.default_rng(21)
        a=cv2.GaussianBlur(rng.normal(128,40,(256,256)).astype(np.float32),(0,0),1.2)
        original=a.copy();matcher=PreciseMatcher(half=24,search=8)
        for shift in ((.25,-.4),(3.25,2.25),(-3.25,-2.25),(.5,.5),(0,0)):
            b=np.fft.ifftn(fourier_shift(np.fft.fftn(a),shift[::-1])).real.astype(np.float32)*1.1+8
            result=matcher.match(SplineImage(a),SplineImage(b),128.37,128.21)
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result[:2],shift,atol=.007)
        np.testing.assert_array_equal(a,original)

    def test_flat_and_border_subsets_rejected(self):
        m=PreciseMatcher();im=SplineImage(np.ones((256,256),np.float32))
        self.assertIsNone(m.match(im,im,128,128))
        self.assertIsNone(m.match(im,im,1,1))

    def test_two_camera_tracking_preserves_nonzero_3d_deformation(self):
        class Stereo:
            epi_thresh=1.5
            def _epi_err(self,x,y,xr,yr):
                return abs(y-yr)
            def _triangulate(self,x,y,xr,yr):
                z=100000/(x-xr)
                return np.column_stack([x*z/1000,y*z/1000,z])
        rng=np.random.default_rng(5)
        a=cv2.GaussianBlur(rng.normal(128,40,(768,768)).astype(np.float32),(0,0),1.3)
        def shifted(im,dx,dy):
            return np.fft.ifftn(fourier_shift(np.fft.fftn(im),(dy,dx))).real.astype(np.float32)
        r=shifted(a,-200,0)
        l1=shifted(a,2.3,-1.1);r1=shifted(r,1.7,-1.1)
        x,y=np.meshgrid(np.linspace(320,650,12),np.linspace(100,650,18))
        x,y=x.ravel(),y.ravel()
        q0=np.column_stack([x,y,x-200,y,np.ones(len(x))])
        e=Stereo();P0=e._triangulate(*q0[:,:4].T)
        ref=dict(correspondence=q0,P=P0,candidates=len(x))
        z=track(e,ref,[SplineImage(a),SplineImage(r)],[SplineImage(l1),SplineImage(r1)],PreciseMatcher(half=24,search=8))
        ids=z['reference_index']
        q1=q0[:,:4]+[2.3,-1.1,1.7,-1.1]
        truth=e._triangulate(*q1.T)-P0
        self.assertGreater(len(z['D']),200)
        self.assertGreater(np.median(abs(z['D'][:,2])),1)
        self.assertLess(np.percentile(abs(z['D']-truth[ids]),95),.02)


if __name__=='__main__':
    unittest.main()
