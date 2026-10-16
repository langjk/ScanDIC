"""Cubic-interpolated local translation with fitted gain/offset; no epipolar constraint."""
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scandic.processing.offline_dic import SubsetMatcher


class SplineImage:
    def __init__(self, image):
        self.image=np.asarray(image,np.float32)
        self.coeff=spline_filter(self.image,order=3,output=np.float32)
        self.shape=self.image.shape

    def sample(self,x,y):
        return map_coordinates(self.coeff,[y,x],order=3,prefilter=False,mode='mirror')


class PreciseMatcher:
    def __init__(self,half=32,search=16,min_ncc=.7,tolerance=1e-4,max_iter=30):
        self.half,self.search,self.min_ncc=half,search,min_ncc
        self.tolerance,self.max_iter=tolerance,max_iter
        self.coarse=SubsetMatcher(half=half,search=search,min_ncc=min_ncc*.8)
        self.ox,self.oy=np.meshgrid(np.arange(-half,half,dtype=float),np.arange(-half,half,dtype=float))

    def match(self,a,b,x,y,init=(0.,0.)):
        coarse=self.coarse.match(a.image,b.image,x,y,init=init)
        if coarse is None:
            return None
        xx,yy=self.ox+x,self.oy+y
        if xx.min()<2 or yy.min()<2 or xx.max()>a.shape[1]-3 or yy.max()>a.shape[0]-3:
            return None
        template=a.sample(xx,yy).astype(np.float64)
        template-=template.mean()
        energy=np.sum(template**2)
        if energy < 1e-8:
            return None
        p=np.array(coarse[:2],float)
        converged=False
        for it in range(self.max_iter):
            tx,ty=xx+p[0],yy+p[1]
            if tx.min()<2 or ty.min()<2 or tx.max()>b.shape[1]-3 or ty.max()>b.shape[0]-3:
                return None
            target=b.sample(tx,ty).astype(np.float64)
            centered=target-target.mean()
            gain=np.sum(template*centered)/energy
            if gain<=0:
                return None
            residual=centered-gain*template
            gy,gx=np.gradient(target)
            # Project gain and offset out of the spatial Jacobian.
            J=np.column_stack([gx.ravel(),gy.ravel()])
            J-=J.mean(0)
            J-=template.ravel()[:,None]*(template.ravel()@J/energy)
            H=J.T@J
            if np.linalg.eigvalsh(H)[0]<1e-6 or np.linalg.cond(H)>1e5:
                return None
            delta=-np.linalg.solve(H,J.T@residual.ravel())
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>3:
                return None
            p+=delta
            if np.linalg.norm(delta)<self.tolerance:
                converged=True
                break
        if not converged:
            return None
        target=b.sample(xx+p[0],yy+p[1]).astype(np.float64)
        target-=target.mean()
        ncc=float(np.sum(template*target)/np.sqrt(energy*np.sum(target**2)))
        if ncc<self.min_ncc or not np.isfinite(ncc):
            return None
        return float(p[0]),float(p[1]),ncc
