"""固定网格的子区匹配核验极线误差，不以标定F筛匹配、不修改标定。"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from dic_engine import DicEngine
from offline_dic import SubsetMatcher


def signed_epipolar(F, l, r):
    lines = np.column_stack([l, np.ones(len(l))]) @ F.T
    return (lines * np.column_stack([r, np.ones(len(r))])).sum(1) / np.linalg.norm(lines[:, :2], axis=1)


def diagnose_pair(eng, left, right, step=128, refine_ecc=False):
    raw = []
    for path in (left, right):
        a = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(str(path))
        raw.append(a.astype(np.float32) / (256 if a.dtype == np.uint16 else 1))
    l, r = eng.undistort_pair(*raw)
    lp, rp = eng._prep(l), eng._prep(r)
    H, ni, nm, pl, pr = eng._bootstrap_H(lp, rp)
    sift_err = signed_epipolar(eng.F, pl, pr)
    rw = cv2.warpPerspective(rp, H, eng.out_size)
    # 只排除无有效右图支持的区域；不利用极线距离筛选。
    support = cv2.warpPerspective(np.ones(r.shape, np.uint8), H, eng.out_size)
    support = cv2.erode(support, np.ones((105, 105), np.uint8)).astype(bool)
    matcher = SubsetMatcher(half=40, search=10, min_ncc=0.8)
    inv = np.linalg.inv(H)
    rows = []
    ecc_rows = []
    for y in range(128, l.shape[0]-128, step):
        for x in range(128, l.shape[1]-128, step):
            if not support[y, x] or l[y-40:y+40, x-40:x+40].std() < 3:
                continue
            m = matcher.match(lp, rw, x, y)
            if m is None:
                continue
            p = inv @ [x + m[0], y + m[1], 1]
            xr, yr = p[:2] / p[2]
            if not (50 < xr < r.shape[1]-50 and 50 < yr < r.shape[0]-50):
                continue
            rows.append([x, y, xr, yr, m[2], m[0], m[1]])
            if refine_ecc:
                try:
                    cc, warp = cv2.findTransformECC(
                        lp[y-40:y+40, x-40:x+40], rw[y-50:y+50, x-50:x+50],
                        np.float32([[1, 0, 10+m[0]], [0, 1, 10+m[1]]]),
                        cv2.MOTION_TRANSLATION,
                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6), None, 1)
                    delta = warp[:, 2]-10
                    if cc >= 0.8 and np.linalg.norm(delta-np.array(m[:2])) < 1.5:
                        q = inv @ [x+delta[0], y+delta[1], 1]
                        ecc_rows.append([x, y, *(q[:2]/q[2]), cc])
                except cv2.error:
                    pass
    rows = np.array(rows)
    if len(rows) < 30:
        raise RuntimeError(f'Only {len(rows)} valid subsets')
    error = signed_epipolar(eng.F, rows[:, :2], rows[:, 2:4])
    summary = {
        'count': len(rows), 'ncc_median': float(np.median(rows[:, 4])),
        'sift_count': ni, 'sift_median_abs_px': float(np.median(abs(sift_err))),
        'median_signed_px': float(np.median(error)),
        'median_abs_px': float(np.median(abs(error))),
        'p95_abs_px': float(np.percentile(abs(error), 95)),
        'robust_spread_px': float(np.median(abs(error-np.median(error)))*1.4826)
    }
    if refine_ecc:
        erows = np.array(ecc_rows)
        if len(erows) < 30:
            raise RuntimeError('Insufficient ECC subsets')
        ee = signed_epipolar(eng.F, erows[:, :2], erows[:, 2:4])
        summary.update(ecc_count=len(erows), ecc_median_abs_px=float(np.median(abs(ee))),
                       ecc_median_signed_px=float(np.median(ee)),
                       ecc_p95_abs_px=float(np.percentile(abs(ee), 95)),
                       ecc_ncc_median=float(np.median(erows[:, 4])))
    return summary, rows, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('--indices', default='0,4,8,12,13')
    ap.add_argument('--ecc', action='store_true')
    args = ap.parse_args()
    d = Path(args.directory)
    meta = json.loads((d/'meta.json').read_text(encoding='utf-8'))
    with np.load(meta['calib']) as c:
        size = tuple(map(int, c['image_size']))
    eng = DicEngine(meta['calib'], out_size=size)
    out = d/('epipolar_diagnostic_ecc' if args.ecc else 'epipolar_diagnostic')
    out.mkdir(exist_ok=True)
    reports = []
    for k in map(int, args.indices.split(',')):
        prefix = 'station' if (d/f'station_{k:02d}_L.png').exists() else 'candidate'
        report, rows, err = diagnose_pair(eng, d/f'{prefix}_{k:02d}_L.png', d/f'{prefix}_{k:02d}_R.png', refine_ecc=args.ecc)
        report['station'] = k
        report['source_prefix'] = prefix
        reports.append(report)
        np.savez(out/f'station_{k:02d}.npz', correspondence=rows, signed_error_px=err)
        print(json.dumps(report), flush=True)
        (out/'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        with Path('WORK_LOG.md').open('a', encoding='utf-8') as fp:
            fp.write(f'\n- {time.strftime("%Y-%m-%d %H:%M:%S")} 固定网格子区复核（未用F筛选）：'+json.dumps(report, ensure_ascii=False)+'。\n')


if __name__ == '__main__':
    main()
