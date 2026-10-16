"""Relative camera-pose correction requires independently known stationary anchors.

Without those anchors, specimen rigid motion and camera motion are observationally
indistinguishable. No automatic stationary-board assumption is made here.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def project(P,K,R=None,t=None):
    Q=np.asarray(P,np.float64)
    if R is not None:
        Q=Q@R.T+np.asarray(t)
    q=Q@K.T
    return q[:,:2]/q[:,2:]


def transform(theta):
    return cv2.Rodrigues(np.asarray(theta[:3],np.float64))[0],np.asarray(theta[3:])


def fit_relative_pose(P,K,q0,q1,anchor_mask):
    if anchor_mask is None:
        raise ValueError('Stationary anchors are required; camera and specimen rigid motion cannot be separated')
    mask=np.asarray(anchor_mask,bool)
    if mask.shape!=(len(P),) or mask.sum()<30:
        raise ValueError('At least 30 stationary anchors required')
    if not np.isfinite(P).all() or not np.isfinite([q0,q1]).all():
        raise ValueError('Nonfinite pose inputs')
    if np.linalg.svd(q0[mask]-q0[mask].mean(0),compute_uv=False)[1]<10:
        raise ValueError('Stationary anchors lack two-dimensional extent')
    P=np.asarray(P,np.float64);base=project(P,K)
    observed=np.asarray(q1)-np.asarray(q0)
    def residual(theta):
        R,t=transform(theta)
        return (project(P[mask],K,R,t)-base[mask]-observed[mask]).ravel()
    result=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=.05,
                         x_scale=np.r_[np.full(3,.001),np.ones(3)],
                         ftol=1e-11,xtol=1e-11,gtol=1e-9,max_nfev=150)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Pose fit did not converge')
    R,t=transform(result.x)
    predicted=project(P,K,R,t)-base
    return dict(theta=result.x,R=R,t=t,predicted=predicted,
                residual=observed-predicted,anchors=int(mask.sum()),cost=float(result.cost))


def anchored_displacement(P,q0,q1,K1,K2,R0,T0,anchor_mask):
    """Reference-correspondence residual is held fixed, not fit to current displacements."""
    P=np.asarray(P,np.float64);T0=np.asarray(T0).reshape(3)
    right0=P@R0.T+T0
    left=fit_relative_pose(P,K1,q0[:,:2],q1[:,:2],anchor_mask)
    right=fit_relative_pose(right0,K2,q0[:,2:4],q1[:,2:4],anchor_mask)
    Rl,tl=left['R'],left['t']
    Rr=right['R']@R0;tr=right['R']@T0+right['t']
    # Fixed errors of the reference stereo reconstruction are common to both
    # times. Correct these fixed projection residuals before differential solve.
    e0=np.column_stack([q0[:,:2]-project(P,K1),q0[:,2:4]-project(right0,K2)])
    q=q1[:,:4]-e0
    A=K1@np.column_stack([Rl,tl]);B=K2@np.column_stack([Rr,tr])
    X=cv2.triangulatePoints(A,B,q[:,:2].T.copy(),q[:,2:4].T.copy())
    X=(X[:3]/X[3]).T
    if not np.isfinite(X).all():
        raise RuntimeError('Pose-compensated triangulation failed')
    relative_R=Rr@Rl.T;relative_T=tr-relative_R@tl
    return X-P,dict(left=left,right=right,relative_R=relative_R,relative_T=relative_T)


def spatial_training_mask(xy,tile=384):
    """Alternating large tiles; a separate half of the field is held out."""
    xy=np.asarray(xy)
    return ((np.floor(xy[:,0]/tile)+np.floor(xy[:,1]/tile)).astype(int)%2)==0
