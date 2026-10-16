"""Reprocess saved scan/static data with cubic two-camera temporal tracking; no hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from dic_engine import DicEngine
from precise_match import PreciseMatcher
from stereo_temporal import views,reference_stereo,track
from scan_pipeline import stats,write_json,log


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--mode',choices=['scan','static'],required=True)
    ap.add_argument('--output',default='precision_results')
    args=ap.parse_args(argv)
    d=args.directory.resolve();m=json.loads((d/'meta.json').read_text(encoding='utf-8'))
    if m['status']!='completed':
        raise ValueError('Acquisition incomplete')
    out=d/args.output;out.mkdir(exist_ok=True)
    if args.mode=='scan':
        ref=Path(m['ref_dir']);prior=d/'scan_results'
        sm=json.loads((prior/'summary.json').read_text(encoding='utf-8'))
        indices=range(len(m['stations']))
    else:
        ref=d;prior=d/'results'
        sm=json.loads((prior/'summary.json').read_text(encoding='utf-8'))
        indices=range(1,len(m['states']))
    files=sorted(set(list(ref.glob('station_[0-9][0-9]_[LR].png'))+list(d.glob('station_[0-9][0-9]_[LR].png'))))
    identity=dict(mode=args.mode,calib_sha256=hashlib.sha256(Path(m['calib']).read_bytes()).hexdigest(),
                  images=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files],
                  code={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                        ('reprocess_precision.py','precise_match.py','stereo_temporal.py','scan_pipeline.py','dic_engine.py','offline_dic.py')})
    identity=json.loads(json.dumps(identity))
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('Output identity changed; use another output directory')
    write_json(manifest,identity)
    write_json(out/'status.json',dict(status='running'))
    cv2.setNumThreads(2)
    with np.load(m['calib']) as c:
        size=tuple(map(int,c['image_size']))
    e=DicEngine(m['calib'],out_size=size);matcher=PreciseMatcher()
    summaries=[];allD=[];allOld=[];allCommonNew=[];allCommonOld=[];static_reference=None;static_a=None
    for k in indices:
        started=time.time();ref_k=k if args.mode=='scan' else 0
        log(out,f'高精度离线处理 {args.mode} {k}：仅使用存盘数据')
        cache=out/f'station_{k:02d}.npz'
        if cache.exists():
            with np.load(cache) as z:
                st={key:z[key] for key in z.files}
        else:
            if args.mode=='static' and static_reference is not None:
                a,r=static_a,static_reference
            else:
                a=views(e,ref,ref_k)
                rc=out/f'reference_{ref_k:02d}.npz'
                if rc.exists():
                    with np.load(rc) as z:
                        r={key:z[key] for key in z.files}
                else:
                    r=reference_stereo(e,a)
                    np.savez_compressed(rc,**r)
                if args.mode=='static':
                    static_a,static_reference=a,r
            b=views(e,d,k)
            st=track(e,r,a,b,matcher)
            np.savez_compressed(cache,**st)
            del b
        if args.mode=='scan':
            basis=np.array(sm['R_to_station0'][k]).T@np.array(sm['basis'])
            old=np.load(prior/f'station_{k:02d}.npz')
        else:
            basis=np.array(sm['basis_local'])
            old=np.load(prior/f'state_{k:02d}.npz')
        newD=st['D']@basis;oldD=old['D']@basis
        oldmap={(int(x),int(y)):i for i,(x,y) in enumerate(zip(old['px'],old['py']))}
        common=[(i,oldmap[(int(x),int(y))]) for i,(x,y) in enumerate(zip(st['px'],st['py'])) if (int(x),int(y)) in oldmap]
        ni,oi=np.array(common).T
        s=dict(index=k,raw=stats(newD),legacy=stats(oldD),common_new=stats(newD[ni]),common_old=stats(oldD[oi]),
               seconds=time.time()-started,reference_count=len(st['P']))
        allD.append(newD);allOld.append(oldD);allCommonNew.append(newD[ni]);allCommonOld.append(oldD[oi])
        summaries.append(s)
        write_json(out/'progress.json',summaries)
        log(out,'U/V/W原始绝对P95='+ '/'.join(f"{s['raw'][c]['p95_abs_um']:.3f}" for c in 'UVW')+f"μm；{len(newD)}点")
    result=dict(mode=args.mode,directory=str(d),states=summaries,raw=stats(np.concatenate(allD)),
                legacy=stats(np.concatenate(allOld)),common_new=stats(np.concatenate(allCommonNew)),
                common_old=stats(np.concatenate(allCommonOld)),
                displacement_correction='none',method='cubic local photometric translation; independent left/right temporal tracking',
                geometry_qualified=bool(m.get('arrival_geometry_passed',True) and m.get('final_geometry_passed',True)))
    write_json(out/'summary.json',result)
    write_json(out/'status.json',dict(status='completed'))
    log(out,'高精度离线处理完成；保留原始零偏，未作相机运动补偿或强制归零')


if __name__=='__main__':
    main()
