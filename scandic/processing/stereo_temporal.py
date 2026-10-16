"""Offline four-image DIC: establish stereo once, independently track both camera images."""
import cv2
import numpy as np
from scandic.processing.precise_match import SplineImage,PreciseMatcher
from scandic.processing.scan_pipeline import load_station,h_apply
from scandic.capture.scan_capture import shift_px


def views(eng,directory,k):
    raw=[load_station(directory,k,t) for t in 'LR']
    if any(im.shape[::-1]!=eng.out_size for im in raw):
        raise ValueError('Input size/calibration mismatch')
    return [SplineImage(im) for im in eng.undistort_pair(*raw)]


def reference_stereo(eng,pair,step=64,subset=64,search=16):
    lp,rp=[eng._prep(v.image) for v in pair]
    H,ni,_,_,_=eng._bootstrap_H(lp,rp)
    a,b=SplineImage(lp),SplineImage(cv2.warpPerspective(rp,H,eng.out_size))
    Hi=np.linalg.inv(H)
    matcher=PreciseMatcher(half=subset//2,search=search,min_ncc=.7)
    image=pair[0].image
    mu=cv2.blur(image,(31,31))
    sd=np.sqrt(np.maximum(cv2.blur(image*image,(31,31))-mu*mu,0))
    texture=(sd>1.5)&(sd/(mu+3)>.1)
    margin=subset//2+search+5
    w,h=eng.out_size
    rows=[];candidates=0
    for y in range(margin,h-margin,step):
        for x in range(margin,w-margin,step):
            if not texture[y,x]:
                continue
            candidates+=1
            m=matcher.match(a,b,x,y)
            if m is None:
                continue
            xr,yr=h_apply(Hi,np.array([x,y])+m[:2])
            if not (margin<xr<w-margin and margin<yr<h-margin):
                continue
            if eng._epi_err(x,y,xr,yr)>eng.epi_thresh:
                continue
            rows.append([x,y,xr,yr,m[2]])
    if len(rows)<200:
        raise RuntimeError(f'Reference stereo only {len(rows)} valid points')
    c=np.array(rows)
    return dict(correspondence=c,P=eng._triangulate(*c[:,:4].T),candidates=np.array(candidates),H=H)


def track(eng,reference,a,b,matcher):
    """Never derive current-right point from a new stereo match against current-left."""
    initial=[]
    for im0,im1 in zip(a,b):
        dx,dy,ncc=shift_px(im0.image,im1.image)
        if not np.isfinite([dx,dy,ncc]).all() or ncc<.4:
            raise RuntimeError('Temporal coarse registration failed')
        initial.append((-dx,-dy))
    ids=[];rows=[];motions=[]
    for i,c in enumerate(reference['correspondence']):
        matches=[matcher.match(a[j],b[j],*c[2*j:2*j+2],init=initial[j]) for j in (0,1)]
        if any(m is None for m in matches):
            continue
        q=np.r_[c[:2]+matches[0][:2],c[2:4]+matches[1][:2]]
        if eng._epi_err(*q)>eng.epi_thresh:
            continue
        ids.append(i);rows.append(q);motions.append(matches)
    if len(ids)<200:
        raise RuntimeError(f'Temporal stereo only {len(ids)} valid points')
    rows=np.asarray(rows);ids=np.array(ids)
    P0=reference['P'][ids]
    P1=eng._triangulate(*rows.T)
    ok=np.isfinite(P0).all(1)&np.isfinite(P1).all(1)&(P0[:,2]>0)&(P1[:,2]>0)
    c=reference['correspondence'][ids][ok]
    return dict(P=P0[ok],D=P1[ok]-P0[ok],px=c[:,0],py=c[:,1],
                ref_correspondence=c,def_correspondence=rows[ok],
                temporal_lr=np.asarray(motions)[ok],reference_index=ids[ok],
                candidates=reference['candidates'])
