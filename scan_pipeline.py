"""全分辨率扫描DIC；保存原始与重叠修正场，不以零位移先验修正数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from dic_engine import DicEngine
from offline_dic import SubsetMatcher
from scan_capture import projective_registration, shift_px

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def log(out, message):
    print(message, flush=True)
    line = f'- {time.strftime("%Y-%m-%d %H:%M:%S")} {message}\n'
    for p in (out / 'processing_log.md', ROOT / 'WORK_LOG.md'):
        with p.open('a', encoding='utf-8') as f:
            f.write(line)


def load_station(d, k, tag):
    p = Path(d) / f'station_{k:02d}_{tag}.png'
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f'图像缺失或无法解码: {p}')
    return im.astype(np.float32) / (256 if im.dtype == np.uint16 else 1)


def h_apply(H, xy):
    xy = np.asarray(xy, np.float64)
    q = np.column_stack([xy.reshape(-1, 2), np.ones(xy.size // 2)]) @ H.T
    return (q[:, :2] / q[:, 2:]).reshape(xy.shape)


def kabsch(A, B):
    """R @ A + t -> B；平面点可用，但共线点不可用。"""
    A, B = np.asarray(A), np.asarray(B)
    if len(A) < 3 or not np.isfinite([A, B]).all():
        raise ValueError('刚体对应不足或非有限')
    ca, cb = A.mean(0), B.mean(0)
    U, sv, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    if sv[1] < 1e-10:
        raise ValueError('共线对应不能确定三维刚体变换')
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    return R, cb - R @ ca


def robust_rigid(A, B, threshold=0.15, minimum=30):
    """固定mm阈值RANSAC；失败不沿用前站。"""
    A, B = np.asarray(A), np.asarray(B)
    if len(A) < minimum or not np.isfinite([A, B]).all():
        raise RuntimeError('站间有限三维对应不足')
    rng = np.random.default_rng(2031)
    best = np.zeros(len(A), bool)
    for _ in range(250):
        ids = rng.choice(len(A), 3, replace=False)
        try:
            R, t = kabsch(A[ids], B[ids])
        except ValueError:
            continue
        ok = np.linalg.norm(A @ R.T + t - B, axis=1) < threshold
        if ok.sum() > best.sum():
            best = ok
    for _ in range(5):
        if best.sum() < max(minimum, 0.5 * len(A)):
            raise RuntimeError(f'站间刚体不一致: {best.sum()}/{len(A)}')
        R, t = kabsch(A[best], B[best])
        res = np.linalg.norm(A @ R.T + t - B, axis=1)
        ok = res < threshold
        if np.array_equal(ok, best):
            break
        best = ok
    if best.sum() < max(minimum, 0.5 * len(A)):
        raise RuntimeError('站间刚体重估失败')
    R, t = kabsch(A[best], B[best])
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    return R, t, best, res


def validate_inputs(def_dir):
    d = json.loads((def_dir / 'meta.json').read_text(encoding='utf-8'))
    ref = Path(d['ref_dir'])
    r = json.loads((ref / 'meta.json').read_text(encoding='utf-8'))
    for meta, role in ((r, 'ref'), (d, 'def')):
        if meta['role'] != role or meta['status'] != 'completed':
            raise ValueError(f'{role}扫描未完整完成')
    if not r['stations'] or r['stations'] != d['stations'] or d['planned_stations'] != r['stations']:
        raise ValueError('两轮站位表不一致')
    if r['avg'] != d['avg'] or r['exposure_us'] != d['exposure_us']:
        raise ValueError('两轮平均数或曝光不一致')
    for directory in (ref, def_dir):
        for k in range(len(r['stations'])):
            for tag in ('L', 'R'):
                if not (directory / f'station_{k:02d}_{tag}.png').is_file():
                    raise ValueError(f'{directory} 站{k}图像缺失')
    a, b = Path(r['calib']).read_bytes(), Path(d['calib']).read_bytes()
    if a != b:
        raise ValueError('两轮绑定标定不一致')
    return ref, r, d, hashlib.sha256(a).hexdigest()


def prepare(eng, directory, k, H=None):
    rawL, rawR = load_station(directory, k, 'L'), load_station(directory, k, 'R')
    if rawL.shape[::-1] != eng.out_size or rawR.shape != rawL.shape:
        raise ValueError('图像尺寸与标定不匹配')
    l, r = eng.undistort_pair(rawL, rawR)
    lp, rp = eng._prep(l), eng._prep(r)
    ni = 0
    if H is None:
        H, ni, nm, _, _ = eng._bootstrap_H(lp, rp)
    mu = cv2.blur(l, (31, 31))
    sd = np.sqrt(np.maximum(cv2.blur(l * l, (31, 31)) - mu * mu, 0))
    texture = (sd > 1.5) & (sd / (mu + 3) > 0.10)
    return dict(lp=lp, rw=cv2.warpPerspective(rp, H, eng.out_size),
                H=H, Hi=np.linalg.inv(H), texture=texture, bootstrap_inliers=ni)


def stereo_point(eng, mat, view, xy):
    m = mat.match(view['lp'], view['rw'], *xy)
    if m is None:
        return None
    qr = h_apply(view['Hi'], np.asarray(xy) + m[:2])
    if not np.isfinite(qr).all() or eng._epi_err(*xy, *qr) > eng.epi_thresh:
        return None
    w, h = eng.out_size
    if not (0 <= qr[0] < w and 0 <= qr[1] < h):
        return None
    return [*xy, *qr, m[2]]


def process_station(eng, mat, a, b, step):
    w, h = eng.out_size
    margin = mat.half + mat.search + 5
    dx, dy, ncc = shift_px(a['lp'], b['lp'])
    init = (-dx, -dy)  # 原点位置与纹理运动相反。
    records0, records1, temporal = [], [], []
    candidates = 0
    for y in range(margin, h - margin, step):
        for x in range(margin, w - margin, step):
            if not a['texture'][y, x]:
                continue
            candidates += 1
            s0 = stereo_point(eng, mat, a, (x, y))
            if s0 is None:
                continue
            mt = mat.match(a['lp'], b['lp'], x, y, init=init)
            if mt is None:
                continue
            s1 = stereo_point(eng, mat, b, (x + mt[0], y + mt[1]))
            if s1 is not None:
                records0.append(s0)
                records1.append(s1)
                temporal.append(mt)
    if len(records0) < 200:
        raise RuntimeError(f'站内有效对应不足: {len(records0)}/{candidates}')
    c0, c1, temp = map(np.asarray, (records0, records1, temporal))
    p0 = eng._triangulate(*c0[:, :4].T)
    p1 = eng._triangulate(*c1[:, :4].T)
    ok = np.isfinite(p0).all(1) & np.isfinite(p1).all(1) & (p0[:, 2] > 0) & (p1[:, 2] > 0)
    if ok.sum() < 200:
        raise RuntimeError('站内有效三维点不足')
    return dict(P=p0[ok], D=p1[ok] - p0[ok], px=c0[ok, 0], py=c0[ok, 1],
                ref_correspondence=c0[ok], def_correspondence=c1[ok], temporal=temp[ok],
                candidates=np.array(candidates), coarse_shift=np.array([dx, dy, ncc]))


def link_stations(eng, mat, a, b, st_a, limit=700):
    H, ncc, ni = projective_registration(a['lp'], b['lp'])
    if not np.isfinite(ncc) or ncc < 0.4:
        raise RuntimeError(f'站间单应配准失败 NCC={ncc}')
    warped = cv2.warpPerspective(b['lp'], H, eng.out_size)
    Hi = np.linalg.inv(H)
    xy = np.column_stack([st_a['px'], st_a['py']])
    qb = h_apply(Hi, xy)
    m = mat.half + mat.search + 8
    w, h = eng.out_size
    ids = np.flatnonzero((qb[:, 0] > m) & (qb[:, 0] < w-m) &
                         (qb[:, 1] > m) & (qb[:, 1] < h-m))
    if len(ids) > limit:
        ids = ids[np.linspace(0, len(ids)-1, limit).astype(int)]
    pairsA, stereoB, pixel_pairs = [], [], []
    for i in ids:
        mt = mat.match(a['lp'], warped, *xy[i])
        if mt is None:
            continue
        actual_b = h_apply(Hi, xy[i] + mt[:2])
        sb = stereo_point(eng, mat, b, actual_b)
        if sb is not None:
            pairsA.append(st_a['P'][i])
            stereoB.append(sb)
            pixel_pairs.append([*xy[i], *actual_b])
    if len(pairsA) < 30:
        raise RuntimeError(f'站间精确三维对应不足: {len(pairsA)}')
    A = np.array(pairsA)
    B = eng._triangulate(*np.array(stereoB)[:, :4].T)
    R, t, inl, res = robust_rigid(B, A)
    report = dict(count=len(A), inliers=int(inl.sum()), ncc=float(ncc), sift_inliers=ni,
                  median_mm=float(np.median(res[inl])), p95_mm=float(np.percentile(res[inl],95)),
                  all_p95_mm=float(np.percentile(res,95)), R=R.tolist(), t=t.tolist(),
                  H_B_to_A=H.tolist(), ransac_threshold_mm=0.15)
    return R, t, report, dict(A=A, B=B, inlier=inl, residual_mm=res, pixel_pairs=pixel_pairs)


def overlap_correction(points, displacements, radius=2.0):
    offsets = [np.zeros(3)]
    seams = []
    for k in range(1, len(points)):
        dist, j = cKDTree(points[k-1]).query(points[k], distance_upper_bound=radius)
        ok = np.isfinite(dist)
        if ok.sum() < 30:
            raise RuntimeError(f'站间{k-1}->{k}重叠不足，不输出伪完整拼接')
        difference = displacements[k][ok] - displacements[k-1][j[ok]]
        offsets.append(offsets[-1] - np.median(difference, axis=0))
        corrected = difference + offsets[-1] - offsets[-2]
        seam = dict(stations=[k-1,k], count=int(ok.sum()),
                    nearest_distance_p95_mm=float(np.percentile(dist[ok],95)))
        for label, value in (('raw',difference),('corrected',corrected)):
            norms = np.linalg.norm(value,axis=1)*1000
            seam[label+'_median_norm_um'] = float(np.median(norms))
            seam[label+'_p95_norm_um'] = float(np.percentile(norms,95))
        seams.append(seam)
    return np.asarray(offsets), seams


def stats(D):
    result = {'count': len(D)}
    for i, name in enumerate('UVW'):
        v = D[:,i] * 1000
        result[name] = dict(median_um=float(np.median(v)), mean_um=float(v.mean()),
                            rms_um=float(np.sqrt(np.mean(v*v))),
                            p95_abs_um=float(np.percentile(np.abs(v),95)))
    result['norm_p95_um'] = float(np.percentile(np.linalg.norm(D,axis=1),95)*1000)
    return result


def plane_basis(P):
    ctr = np.mean(P,axis=0)
    _, _, vt = np.linalg.svd(P[::max(1,len(P)//8000)] - ctr, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    e1 = np.array([1.,0,0]) - n[0]*n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n,e1)
    return ctr, np.column_stack([e1,e2,n])


def main(argv=None):
    from scan_report import plot_results, write_report
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('def_dir',type=Path)
    ap.add_argument('--step',type=int,default=64)
    ap.add_argument('--subset',type=int,default=64)
    ap.add_argument('--search',type=int,default=16)
    ap.add_argument('--grid-mm',type=float,default=1.0)
    ap.add_argument('--output',default='scan_results')
    args = ap.parse_args(argv)
    if args.step < 1 or args.subset < 8 or args.subset % 2 or args.search < 0 or not np.isfinite(args.grid_mm) or args.grid_mm <= 0:
        ap.error('参数必须有效，subset必须为偶数')
    cv2.setNumThreads(4)
    def_dir=args.def_dir.resolve()
    ref, rm, dm, chash=validate_inputs(def_dir)
    out=def_dir/args.output
    out.mkdir(parents=True,exist_ok=True)
    params=dict(step=args.step,subset=args.subset,search=args.search,grid_mm=args.grid_mm)
    identity=dict(parameters=params,calibration_sha256=chash,
                  code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ('scan_pipeline.py','scan_report.py','offline_dic.py','dic_engine.py','stitch.py')},
                  input_images=[(str(p),p.stat().st_size,p.stat().st_mtime_ns)
                                for d in (ref,def_dir) for p in sorted(d.glob('station_[0-9][0-9]_*.png'))])
    identity=json.loads(json.dumps(identity))
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError('输出目录有不同配置/输入/代码的结果，请用 --output 新目录')
    write_json(manifest,identity)
    write_json(out/'status.json',dict(status='running',time=time.strftime('%Y-%m-%d %H:%M:%S')))
    try:
        with np.load(dm['calib']) as c:
            size=tuple(map(int,c['image_size']))
        eng=DicEngine(dm['calib'],out_size=size)
        mat=SubsetMatcher(half=args.subset//2,search=args.search,min_ncc=0.7)
        Rs,ts=[np.eye(3)],[np.zeros(3)]
        stations,links=[],[]
        previous=None
        for k in range(len(dm['stations'])):
            log(out,f'零位移处理站{k}：读取与立体预配准')
            current=prepare(eng,ref,k)
            cache=out/f'station_{k:02d}.npz'
            if cache.exists():
                with np.load(cache) as z:
                    st={key:z[key] for key in z.files}
            else:
                # 同站两时刻共用预扭映射，避免更换插值网格引入时序偏差。
                deformed=prepare(eng,def_dir,k,H=current['H'])
                st=process_station(eng,mat,current,deformed,args.step)
                del deformed
                np.savez_compressed(cache,**st)
            stations.append(st)
            log(out,f"站{k}有效点{len(st['P'])}/{int(st['candidates'])}，原始|D|中位{np.median(np.linalg.norm(st['D'],axis=1))*1000:.2f}μm")
            if k:
                R,t,rep,correspondence=link_stations(eng,mat,previous,current,stations[k-1])
                np.savez_compressed(out/f'link_{k-1:02d}_{k:02d}.npz',**correspondence)
                links.append(rep)
                ts.append(Rs[-1]@t+ts[-1])
                Rs.append(Rs[-1]@R)
                write_json(out/'links.json',links)
                log(out,f"站间{k-1}→{k}：{rep['inliers']}/{rep['count']}，三维残差P95={rep['p95_mm']:.4f}mm")
            previous=current
        P=[s['P']@R.T+t for s,R,t in zip(stations,Rs,ts)]
        D=[s['D']@R.T for s,R in zip(stations,Rs)]
        offsets,seams=overlap_correction(P,D)
        allP=np.concatenate(P)
        raw=np.concatenate(D)
        corrected=np.concatenate([d+c for d,c in zip(D,offsets)])
        K=np.concatenate([np.full(len(p),k) for k,p in enumerate(P)])
        ctr,basis=plane_basis(allP)
        coords=(allP-ctr)@basis
        uvw_raw,uvw_corrected=raw@basis,corrected@basis
        report=dict(ref_dir=str(ref),def_dir=str(def_dir),parameters=params,calibration_sha256=chash,
                    coverage_mm=np.ptp(coords[:,:2],axis=0).tolist(),
                    raw=stats(uvw_raw),corrected=stats(uvw_corrected),links=links,seams=seams,
                    correction_camera0_mm=offsets.tolist(),R_to_station0=np.asarray(Rs).tolist(),
                    t_to_station0_mm=np.asarray(ts).tolist(),basis=basis.tolist(),
                    stations=[dict(raw=stats(uvw_raw[K==k]),corrected=stats(uvw_corrected[K==k])) for k in range(len(stations))])
        grid={}
        ij=np.floor((coords[:,:2]-coords[:,:2].min(0))/args.grid_mm).astype(int)
        shape=tuple((ij.max(0)+1)[::-1])
        count=np.zeros(shape,int)
        np.add.at(count,(ij[:,1],ij[:,0]),1)
        for label,values in (('raw',uvw_raw),('corrected',uvw_corrected)):
            for i,name in enumerate('UVW'):
                acc=np.zeros(shape)
                np.add.at(acc,(ij[:,1],ij[:,0]),values[:,i])
                grid[f'{label}_{name}']=np.where(count>0,acc/np.maximum(count,1),np.nan)
        np.savez_compressed(out/'merged.npz',P=allP,D_raw=raw,D_corrected=corrected,station=K,
                            coordinates=coords,uvw_raw=uvw_raw,uvw_corrected=uvw_corrected,
                            correction=offsets,basis=basis,center=ctr,grid_count=count,**grid)
        write_json(out/'summary.json',report)
        plot_results(out,coords[:,:2],uvw_raw,uvw_corrected,K,report)
        write_report(out,report)
        write_json(out/'status.json',dict(status='completed'))
        log(out,f'零位移处理完成：{len(allP)}点；报告 {out / "zero_validation.md"}')
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',error=str(exc)))
        log(out,f'零位移处理中止：{exc}')
        raise
