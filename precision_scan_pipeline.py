"""Production offline scan processing: cubic four-image DIC, explicit stationary anchors.

Unanchored fields are apparent displacement in camera coordinates. Overlap-only
offsets are diagnostic outputs, never the default measured displacement.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from camera_motion import anchored_displacement
from dic_engine import DicEngine
from precise_match import PreciseMatcher,SplineImage
from stereo_temporal import views,reference_stereo,track
from scan_pipeline import (validate_inputs,prepare,link_stations,plane_basis,
                           overlap_correction,stats,write_json,log)

ROOT=Path(__file__).resolve().parent


class CachedMatcher:
    def __init__(self,subset=64,search=16):
        self.half,self.search=subset//2,search
        self.matcher=PreciseMatcher(half=self.half,search=search)
        self.cache=OrderedDict()

    def image(self,array):
        key=id(array)
        if key not in self.cache:
            self.cache[key]=SplineImage(array)
        self.cache.move_to_end(key)
        while len(self.cache)>4:
            self.cache.popitem(last=False)
        return self.cache[key]

    def match(self,a,b,x,y,init=(0.,0.)):
        return self.matcher.match(self.image(a),self.image(b),x,y,init)


def verified_cache(cache,ref,d,calib_sha,subset,step,search):
    if (subset,step,search)!=(64,64,16):
        raise ValueError('Existing precision cache was generated with 64/64/16')
    m=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    expected=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in
              sorted(set(list(ref.glob('station_[0-9][0-9]_[LR].png'))+list(d.glob('station_[0-9][0-9]_[LR].png'))))]
    if (m['mode']!='scan' or m['calib_sha256']!=calib_sha or m['images']!=json.loads(json.dumps(expected))):
        raise ValueError('Precision cache input/calibration mismatch')
    for name in ('precise_match.py','stereo_temporal.py'):
        if m['code'][name]!=hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
            raise ValueError('Precision cache matching algorithm changed')
    if json.loads((cache/'status.json').read_text())['status']!='completed':
        raise ValueError('Precision cache is incomplete')


def anchor_mask(config,k,xy):
    if config is None:
        return None
    if config.get('known_stationary') is not True:
        raise ValueError('Anchor configuration must explicitly declare known_stationary=true')
    boxes=config.get('stationary_regions',{}).get(str(k),[])
    if not boxes:
        raise ValueError(f'No independently stationary anchor regions for station {k}')
    mask=np.zeros(len(xy),bool)
    for box in boxes:
        if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid stationary anchor rectangle')
        x0,y0,x1,y1=box
        mask|=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
    return mask


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('def_dir',type=Path)
    ap.add_argument('--output',default='precision_scan_results')
    ap.add_argument('--step',type=int,default=64)
    ap.add_argument('--subset',type=int,default=64)
    ap.add_argument('--search',type=int,default=16)
    ap.add_argument('--grid-mm',type=float,default=1.)
    ap.add_argument('--cache',type=Path,help='Verified reprocess_precision scan cache')
    ap.add_argument('--anchors',type=Path,help='Explicit independently stationary regions in undistorted reference-left pixel coordinates')
    ap.add_argument('--overlap-diagnostic',action='store_true')
    args=ap.parse_args(argv)
    if (args.step<1 or args.subset<8 or args.subset%2 or args.search<0 or
            not np.isfinite(args.grid_mm) or args.grid_mm<=0):
        ap.error('Invalid sampling parameters')
    d=args.def_dir.resolve();ref,rm,dm,chash=validate_inputs(d)
    if args.cache:
        args.cache=args.cache.resolve()
        verified_cache(args.cache,ref,d,chash,args.subset,args.step,args.search)
    anchors=json.loads(args.anchors.read_text(encoding='utf-8')) if args.anchors else None
    out=d/args.output
    if out.exists():
        raise ValueError('Output already exists; use a new --output directory to preserve prior results')
    out.mkdir(parents=True)
    write_json(out/'status.json',dict(status='running'))
    write_json(out/'manifest.json',dict(calibration_sha256=chash,
               parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               anchors=anchors,
               code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}))
    cv2.setNumThreads(2)
    with np.load(dm['calib']) as c:
        e=DicEngine(dm['calib'],out_size=tuple(map(int,c['image_size'])))
    precise=PreciseMatcher(half=args.subset//2,search=args.search)
    stations=[];links=[];rotations=[np.eye(3)];translations=[np.zeros(3)];previous=None
    try:
        for k in range(len(dm['stations'])):
            log(out,f'精确扫描拼接站{k}')
            if args.cache:
                with np.load(args.cache/f'station_{k:02d}.npz') as z:
                    st={key:z[key] for key in z.files}
            else:
                a,b=views(e,ref,k),views(e,d,k)
                reference=reference_stereo(e,a,step=args.step,subset=args.subset,search=args.search)
                st=track(e,reference,a,b,precise)
                del a,b
            st['D_output']=st['D'].copy()
            if anchors is not None:
                mask=anchor_mask(anchors,k,st['ref_correspondence'][:,:2])
                corrected,pose=anchored_displacement(st['P'],st['ref_correspondence'],st['def_correspondence'],e.K1,e.K2,e.R,e.T,mask)
                st['D_output']=corrected;st['anchor_mask']=mask
                st['camera_left_pose']=pose['left']['theta'];st['camera_right_pose']=pose['right']['theta']
            np.savez_compressed(out/f'station_{k:02d}.npz',**st)
            current=prepare(e,ref,k)
            if k:
                R,t,rep,pairs=link_stations(e,CachedMatcher(args.subset,args.search),previous,current,stations[-1])
                translations.append(rotations[-1]@t+translations[-1]);rotations.append(rotations[-1]@R)
                links.append(rep);np.savez_compressed(out/f'link_{k-1:02d}_{k:02d}.npz',**pairs)
                write_json(out/'links.json',links)
                log(out,f"连接{k-1}→{k}：{rep['inliers']}/{rep['count']}，精确三维残差P95={rep['p95_mm']:.5f}mm")
            stations.append(st);previous=current
        P=[s['P']@R.T+t for s,R,t in zip(stations,rotations,translations)]
        raw=[s['D']@R.T for s,R in zip(stations,rotations)]
        fields=[s['D_output']@R.T for s,R in zip(stations,rotations)]
        allP=np.concatenate(P);allRaw=np.concatenate(raw);allD=np.concatenate(fields)
        station=np.concatenate([np.full(len(p),k) for k,p in enumerate(P)])
        center,basis=plane_basis(allP);coords=(allP-center)@basis
        uvw=allD@basis;rawuvw=allRaw@basis
        data=dict(P=allP,D_raw=allRaw,D=allD,station=station,coordinates=coords,
                  uvw_raw=rawuvw,uvw=uvw,basis=basis,center=center,
                  rotations=rotations,translations=translations)
        summary=dict(absolute_displacement_identifiable=anchors is not None,
                     output_quantity='anchored displacement' if anchors else 'apparent displacement; camera/specimen rigid motion not separable',
                     correction='stationary anchors' if anchors else 'none',raw=stats(rawuvw),output=stats(uvw),
                     coverage_mm=np.ptp(coords[:,:2],axis=0).tolist(),links=links,
                     stations=[stats(uvw[station==k]) for k in range(len(stations))])
        if args.overlap_diagnostic:
            offsets,seams=overlap_correction(P,raw)
            diagnostic=np.concatenate([field+offset for field,offset in zip(raw,offsets)])@basis
            data['uvw_overlap_diagnostic']=diagnostic
            data['overlap_diagnostic_offsets']=offsets
            summary['overlap_diagnostic']=stats(diagnostic);summary['overlap_diagnostic_seams']=seams
        ij=np.floor((coords[:,:2]-coords[:,:2].min(0))/args.grid_mm).astype(int)
        shape=tuple((ij.max(0)+1)[::-1]);counts=np.zeros(shape,int)
        np.add.at(counts,(ij[:,1],ij[:,0]),1)
        data['grid_count']=counts;data['grid_mm']=args.grid_mm
        for j,c in enumerate('UVW'):
            accum=np.zeros(shape);np.add.at(accum,(ij[:,1],ij[:,0]),uvw[:,j])
            data['grid_'+c]=np.where(counts>0,accum/np.maximum(counts,1),np.nan)
        np.savez_compressed(out/'merged.npz',**data)
        write_json(out/'summary.json',summary)
        lines=['# 精确扫描DIC处理','',f"输出：{summary['output_quantity']}；相机补偿：{summary['correction']}。",'',
               '没有独立静止锚点时，原始整体偏差保留；不自动用重叠平移链或逐站归零代替真实位移。','',
               '|分量|中位μm|RMS μm|绝对P95 μm|','|---|---:|---:|---:|']
        for c in 'UVW':
            s=summary['output'][c];lines.append(f"|{c}|{s['median_um']:.3f}|{s['rms_um']:.3f}|{s['p95_abs_um']:.3f}|")
        lines+=['','旧扫描结果、旧标定和原始图像均未修改。详细三维连接、对应点、运行参数与源码哈希在本目录保存。','']
        (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        write_json(out/'status.json',dict(status='completed'))
        log(out,'精确扫描DIC处理完成：'+str(out/'report.md'))
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':
    main()
