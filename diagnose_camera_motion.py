"""Known-zero experiment only: fit camera motion on spatial train tiles, evaluate held-out tiles."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from camera_motion import anchored_displacement,spatial_training_mask,project,transform
from dic_engine import DicEngine
from scan_pipeline import stats,write_json


def rigid_rig_displacement(P,q0,q1,e,train):
    """Competing six-DOF model: stereo R/T stays fixed, only the whole rig moves."""
    T=e.T.reshape(3)
    baseline=np.column_stack([project(P,e.K1),project(P,e.K2,e.R,T)])
    observed=q1[:,:4]-q0[:,:4]
    def prediction(theta):
        R,t=transform(theta)
        moved=P@R.T+t
        return np.column_stack([project(moved,e.K1),project(moved,e.K2,e.R,T)])-baseline
    result=least_squares(lambda theta:(prediction(theta)[train]-observed[train]).ravel(),
                         np.zeros(6),loss='soft_l1',f_scale=.05,
                         x_scale=np.r_[np.full(3,.001),np.ones(3)],
                         ftol=1e-11,xtol=1e-11,gtol=1e-9,max_nfev=150)
    if not result.success:
        raise RuntimeError('Rigid rig diagnostic did not converge')
    R,t=transform(result.x)
    corrected=q1[:,:4]-(q0[:,:4]-baseline)
    A=e.K1@np.column_stack([R,t]);B=e.K2@np.column_stack([e.R@R,e.R@t+T])
    X=cv2.triangulatePoints(A,B,corrected[:,:2].T.copy(),corrected[:,2:4].T.copy())
    return (X[:3]/X[3]).T-P,result.x


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--known-stationary-zero-test',action='store_true',required=True)
    ap.add_argument('--indices',default=None)
    args=ap.parse_args(argv)
    d=args.directory.resolve();m=json.loads((d/'meta.json').read_text(encoding='utf-8'))
    if m.get('experiment',{}).get('type')!='unloaded_zero_displacement_validation':
        raise ValueError('This diagnostic requires an explicitly recorded unloaded zero-displacement experiment')
    source=d/'precision_results';out=source/'camera_motion_diagnostic';out.mkdir(exist_ok=True)
    basisinfo=json.loads((d/'scan_results'/'summary.json').read_text(encoding='utf-8'))
    with np.load(m['calib']) as c:
        e=DicEngine(m['calib'],out_size=tuple(map(int,c['image_size'])))
    ks=list(map(int,args.indices.split(','))) if args.indices else range(len(m['stations']))
    rows=[];raw_test=[];corrected_test=[];rigid_test=[]
    for k in ks:
        with np.load(source/f'station_{k:02d}.npz') as saved:
            z={key:saved[key] for key in saved.files}
        train=spatial_training_mask(z['ref_correspondence'][:,:2])
        D,pose=anchored_displacement(z['P'],z['ref_correspondence'],z['def_correspondence'],e.K1,e.K2,e.R,e.T,train)
        basis=np.array(basisinfo['R_to_station0'][k]).T@np.array(basisinfo['basis'])
        test=~train
        raw=z['D']@basis;corrected=D@basis
        rigidD,rigidtheta=rigid_rig_displacement(z['P'],z['ref_correspondence'],z['def_correspondence'],e,train)
        rigiduvw=rigidD@basis
        raw_test.append(raw[test]);corrected_test.append(corrected[test])
        rigid_test.append(rigiduvw[test])
        row=dict(station=k,train_count=int(train.sum()),test_count=int(test.sum()),
                 raw_test=stats(raw[test]),corrected_test=stats(corrected[test]),corrected_train=stats(corrected[train]),
                 rigid_rig_test=stats(rigiduvw[test]),rigid_rig_pose=rigidtheta.tolist(),
                 left_pose=pose['left']['theta'].tolist(),right_pose=pose['right']['theta'].tolist(),
                 relative_stereo_rotation_change_deg=float(np.linalg.norm(cv2.Rodrigues(pose['relative_R']@e.R.T)[0])*180/np.pi),
                 relative_baseline_change_mm=float(np.linalg.norm(pose['relative_T'])-np.linalg.norm(e.T)),
                 heldout_left_residual_p95_px=float(np.percentile(np.linalg.norm(pose['left']['residual'][test],axis=1),95)),
                 heldout_right_residual_p95_px=float(np.percentile(np.linalg.norm(pose['right']['residual'][test],axis=1),95)))
        rows.append(row)
        np.savez_compressed(out/f'station_{k:02d}.npz',train_mask=train,D_anchored=D,uvw_anchored=corrected,
                            D_rigid_rig=rigidD,uvw_rigid_rig=rigiduvw,
                            left_pose=pose['left']['theta'],right_pose=pose['right']['theta'])
        print(k,'heldout P95',*[round(row['corrected_test'][c]['p95_abs_um'],3) for c in 'UVW'],
              'relative stereo rotation deg',row['relative_stereo_rotation_change_deg'],flush=True)
    write_json(out/'summary.json',dict(stationary_scene_assumed=True,production_correction=False,
              evaluation='alternating 384px spatial tiles; held-out data excluded from fitting',
              caveat='Existing zero test identifies a correction under a known stationary scene; a loaded experiment requires independent stationary anchors.',
              raw_test=stats(np.concatenate(raw_test)),corrected_test=stats(np.concatenate(corrected_test)),
              rigid_rig_test=stats(np.concatenate(rigid_test)),stations=rows))


if __name__=='__main__':
    main()
