"""复用扫描DIC匹配器处理固定站重复图像；不扣整体平移，不做重叠修正。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scandic.dic_engine import DicEngine
from scandic.processing.offline_dic import SubsetMatcher
from scandic.processing.scan_pipeline import prepare, process_station, stats, write_json, log

from scandic import paths


def verified_static_cache(cache,d,calib):
    manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    files=sorted(d.glob('station_[0-9][0-9]_[LR].png'))
    expected=[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in files]
    if (manifest['mode']!='static' or manifest['images']!=expected or
            manifest['calib_sha256']!=hashlib.sha256(Path(calib).read_bytes()).hexdigest()):
        raise ValueError('静止缓存输入或标定不一致')
    for name in ('precise_match.py','stereo_temporal.py'):
        if manifest['code'][name]!=paths.hash_named([name])[name]:
            raise ValueError('静止缓存匹配算法已改变')
    if json.loads((cache/'status.json').read_text())['status']!='completed':
        raise ValueError('静止缓存未完成')


def common_cube(states, fields):
    maps=[{(int(x),int(y)):i for i,(x,y) in enumerate(zip(st['px'],st['py']))} for st in states]
    common=set(maps[0])
    for mapping in maps[1:]:
        common.intersection_update(mapping)
    points=sorted(common,key=lambda q:(q[1],q[0]))
    if len(points)<200:
        raise RuntimeError('共同有效点不足，不能可靠比较静止时序')
    cube=np.asarray([d[[mapping[q] for q in points]] for d,mapping in zip(fields,maps)])
    return np.asarray(points),cube


def plots(out, times, cube, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,name in enumerate('UVW'):
        med=np.median(cube[:,:,i],axis=1)*1000
        p95=np.percentile(abs(cube[:,:,i]),95,axis=1)*1000
        axes[0,i].plot(times,med,'o-')
        axes[0,i].axhline(0,color='gray',lw=.8)
        axes[0,i].set(title=f'{name}: median, common points',xlabel='Time from reference (s)',ylabel='um')
        axes[1,i].plot(times,p95,'o-',label='Stationary repeat')
        axes[1,i].axhline(summary['scan_station'][name]['p95_abs_um'],color='tab:red',ls='--',label=f"Two scans, station {summary['target_station']}")
        axes[1,i].set(title=f'{name}: absolute P95 (log scale)',xlabel='Time from reference (s)',ylabel='um',yscale='log')
        axes[1,i].legend()
    fig.savefig(out/'static_time_series.png',dpi=160)
    plt.close(fig)


def report_md(out, report, meta):
    k=meta['target_station']
    lines=[f'# 站{k}不运动重复采集：实测结果','',
           f'试验先移动至站{k}，稳定后连续采集；状态0为新的静止参考，后续状态均与它比较。未扣除整体位移、未逐帧归零、未做站间重叠修正。','',
           f"- 数据：`{report['directory']}`。",
           f"- {report['state_count']}组双目图，{report['comparison_count']}组相对参考的位移场；每组{meta['avg']}帧平均，曝光{meta['exposure_us']}us；匹配器：{report['matcher']}。",
           f"- 时序覆盖{report['duration_s']:.2f}s，共同有效点{report['common_points']}；全分辨率、64px子区、64px采样、16px搜索、NCC≥0.7，复用扫描DIC实现。",
           f'- U/V/W采用此前扫描站{k}变换到共同试件系的方向，便于同站比较。统计单位μm。',
           f"- 历史采集阶段编码器跨度{report['encoder_span_deg']:.6f}°；重复采集期间未发运动命令。本次处理仅读取存盘文件。",'',
           '|数据|分量|中位|RMS|绝对P95|','|---|---|---:|---:|---:|']
    if not report['geometry_passed']:
        lines[2:2]=['**几何闸门未通过：本报告仅为诊断结果。三维数值沿用原标定换算，不能视为有效标定下的计量重复性验证。未降低0.8px闸门或更新标定。**','']
    for key,label in (('pooled','本次静止：全部有效点'),('common_pooled','本次静止：共同点'),('scan_station',f'此前两轮扫描：站{k}原始场')):
        for name in 'UVW':
            s=report[key][name]
            lines.append(f"|{label}|{name}|{s['median_um']:+.3f}|{s['rms_um']:.3f}|{s['p95_abs_um']:.3f}|")
    lines+=['','## 时序统计','',f"逐点时间标准差在{report['comparison_count']}个比较状态上计算，包含观测期间漂移；共有参考帧误差在时间差分中是共同项，不能用该标准差代替绝对准确度。",'',
            '|分量|逐点时间SD中位μm|逐点时间SD P95 μm|空间中位随时间线性斜率 μm/s|',
            '|---|---:|---:|---:|']
    for name in 'UVW':
        s=report['temporal'][name]
        lines.append(f"|{name}|{s['point_std_median_um']:.3f}|{s['point_std_p95_um']:.3f}|{s['median_slope_um_per_s']:+.5f}|")
    lines+=['','## 每个状态','','|状态|参考后时间s|有效点|U中位μm|V中位μm|W中位μm|U绝对P95|V绝对P95|W绝对P95|',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in report['states']:
        values=[s['stats'][name]['median_um'] for name in 'UVW']+[s['stats'][name]['p95_abs_um'] for name in 'UVW']
        lines.append(f"|{s['index']}|{s['time_s']:.2f}|{s['stats']['count']}|"+'|'.join(f'{v:.3f}' for v in values)+'|')
    lines+=['','## 采集几何与结论边界','',
            f"- 到站几何核验：`{json.dumps(meta['arrival_geometry'],ensure_ascii=False)}`。",
            f"- 采集后几何核验：`{json.dumps(meta['final_geometry'],ensure_ascii=False)}`。",
            '- 本次仅测试一个站位、约一分钟内的不运动重复性；不等于全部站位、长时间或运动后的性能。',
            '- 与此前两轮扫描相比，时间间隔与参考采集时刻也发生变化，因此不能把误差差值全部归因于滑台机械运动。',
            '- 该比较未改变散斑、曝光或标定；它可判断现有成像条件能否取得较低的静止噪声，但不能独立证明散斑是最优或定位具体硬件故障。',
            '- PNG解码、尺寸/类型与SHA256验证见data_verification.json；逐状态三维点、原始位移与对应点均保留。','',
            '![静止时序及扫描对照](static_time_series.png)','']
    (out/'static_validation.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--matcher',choices=['precise','legacy'],default='precise')
    ap.add_argument('--output',default='precision_static_results')
    ap.add_argument('--cache',type=Path,help='复用经过身份核验的 reprocess_precision 静止缓存')
    ap.add_argument('--diagnostic',action='store_true',help='允许处理几何闸门失败的数据，报告仍标为仅供诊断')
    args=ap.parse_args(argv)
    d=args.directory.resolve()
    meta=json.loads((d/'meta.json').read_text(encoding='utf-8'))
    if (meta.get('role')!='static_station' or meta['status']!='completed' or
            len(meta['states'])!=meta['planned_states'] or len(meta['states'])<3):
        raise ValueError('固定站采集不完整')
    if not (meta['arrival_geometry_passed'] and meta['final_geometry_passed']) and not args.diagnostic:
        raise ValueError('几何闸门未通过；仅允许显式 --diagnostic 处理，不能作正常计量结果')
    if args.cache:
        if args.matcher!='precise':
            raise ValueError('静止缓存只支持 precise 匹配器')
        args.cache=args.cache.resolve()
        verified_static_cache(args.cache,d,meta['calib'])
    out=d/args.output
    if out.exists():
        raise ValueError('输出目录已存在，请另选 --output 保留已有结果')
    out.mkdir()
    source=Path(meta['source_scan'])
    comparison_file=source/'precision_scan_results'/'summary.json'
    if not comparison_file.exists():
        comparison_file=source/'scan_results'/'summary.json'
    comparison=json.loads(comparison_file.read_text(encoding='utf-8'))
    scan_meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    if Path(meta['calib']).read_bytes()!=Path(scan_meta['calib']).read_bytes():
        raise ValueError('静止序列与扫描对照的标定不一致')
    k=meta['target_station']
    if 'R_to_station0' in comparison:
        R=np.array(comparison['R_to_station0'][k])
        basis=R.T@np.array(comparison['basis'])
        scan_station=comparison['stations'][k]['raw']
    else:
        with np.load(comparison_file.parent/'merged.npz') as z:
            basis=z['rotations'][k].T@z['basis']
            scan_station=stats(z['uvw_raw'][z['station']==k])
    cv2.setNumThreads(4)
    with np.load(meta['calib']) as c:
        full=tuple(map(int,c['image_size']))
    verified=[]
    for i in range(len(meta['states'])):
        for tag in 'LR':
            p=d/f'station_{i:02d}_{tag}.png'
            im=cv2.imread(str(p),-1)
            if im is None or im.shape[::-1]!=full or im.dtype!=np.uint16:
                raise RuntimeError(f'图像损坏: {p}')
            verified.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(out/'data_verification.json',dict(count=len(verified),images=verified))
    write_json(out/'manifest.json',dict(parameters=dict(subset=64,step=64,search=16,min_ncc=.7),
              calibration_sha256=hashlib.sha256(Path(meta['calib']).read_bytes()).hexdigest(),
              matcher=args.matcher,
              source_summary_sha256=hashlib.sha256(comparison_file.read_bytes()).hexdigest(),
              cache=str(args.cache) if args.cache else None,
              code_sha256=paths.hash_named(('process_static_station.py','scan_pipeline.py','offline_dic.py','dic_engine.py','stitch.py','precise_match.py','stereo_temporal.py'))))
    eng=DicEngine(meta['calib'],out_size=full)
    if args.matcher=='precise' and not args.cache:
        from scandic.processing.precise_match import PreciseMatcher
        from scandic.processing.stereo_temporal import views,reference_stereo,track
        mat=PreciseMatcher(half=32,search=16,min_ncc=.7)
        original_views=views(eng,d,0)
        ref=reference_stereo(eng,original_views)
    elif args.matcher=='legacy':
        mat=SubsetMatcher(half=32,search=16,min_ncc=.7)
        ref=prepare(eng,d,0)
    states=[];fields=[];summaries=[];times=[]
    reference_time=meta['states'][0]['timestamp_unix']
    for i in range(1,len(meta['states'])):
        if args.cache:
            with np.load(args.cache/f'station_{i:02d}.npz') as z:
                st={key:z[key] for key in z.files}
        elif args.matcher=='precise':
            current=views(eng,d,i)
            st=track(eng,ref,original_views,current,mat)
        else:
            current=prepare(eng,d,i,H=ref['H'])
            st=process_station(eng,mat,ref,current,64)
        D=st['D']@basis
        np.savez_compressed(out/f'state_{i:02d}.npz',**st,uvw=D)
        states.append(st);fields.append(D)
        elapsed=meta['states'][i]['timestamp_unix']-reference_time
        times.append(elapsed)
        s=stats(D)
        summaries.append(dict(index=i,time_s=elapsed,stats=s,
                              temporal_ncc_median=float(np.median(st['temporal_lr'][:,0,2] if args.matcher=='precise' else st['temporal'][:,2]))))
        write_json(out/'progress.json',summaries)
        log(out,f"固定站状态{i}/{len(meta['states'])-1}：{len(D)}点；U/V/W绝对P95="+
            '/'.join(f"{s[name]['p95_abs_um']:.3f}" for name in 'UVW')+'μm')
    points,cube=common_cube(states,fields)
    times=np.asarray(times)
    sd=np.std(cube,axis=0,ddof=1)*1000
    temporal={name:dict(point_std_median_um=float(np.median(sd[:,j])),
                       point_std_p95_um=float(np.percentile(sd[:,j],95)),
                       median_slope_um_per_s=float(np.polyfit(times,np.median(cube[:,:,j],axis=1)*1000,1)[0]))
              for j,name in enumerate('UVW')}
    enc=[s[key] for s in meta['states'] for key in ('encoder_before_deg','encoder_after_deg')]
    report=dict(directory=str(d),target_station=k,state_count=len(meta['states']),comparison_count=len(states),
                geometry_passed=meta['arrival_geometry_passed'] and meta['final_geometry_passed'],
                common_points=len(points),duration_s=float(times[-1]),encoder_span_deg=float(np.ptp(enc)),
                pooled=stats(np.concatenate(fields)),common_pooled=stats(cube.reshape(-1,3)),
                scan_station=scan_station,temporal=temporal,states=summaries,matcher=args.matcher,
                basis_local=basis.tolist())
    np.savez_compressed(out/'common_time_series.npz',reference_pixels=points,uvw=cube,time_s=times,basis_local=basis)
    write_json(out/'summary.json',report)
    plots(out,times,cube,report)
    report_md(out,report,meta)
    log(out,'固定站不运动验证完成：'+str(out/'static_validation.md'))


if __name__=='__main__':
    main()
