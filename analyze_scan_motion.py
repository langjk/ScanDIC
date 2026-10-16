"""用四幅已保存双目图诊断站间相机运动，不操作硬件。单位mm。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dic_engine import contrast_normalize


def features(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(str(path))
    f = image.astype(np.float32) / (256 if image.dtype == np.uint16 else 1)
    f = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    key, desc = cv2.SIFT_create(nfeatures=12000).detectAndCompute(
        contrast_normalize(f).astype(np.uint8), None)
    return np.float32([k.pt for k in key]) * 2, desc


def links(a, b):
    pairs = cv2.BFMatcher().knnMatch(a[1], b[1], k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < 0.7 * n.distance]
    src = np.float32([a[0][m.queryIdx] for m in good])
    dst = np.float32([b[0][m.trainIdx] for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError('Homography failed')
    return {m.queryIdx: m.trainIdx for m, ok in zip(good, mask.ravel()) if ok}, H


def stereo(left, right, matches, c):
    ids = np.array(list(matches))
    l = cv2.undistortPoints(left[0][ids, None], c['K1'], c['D1'], P=c['K1'])[:, 0]
    r = cv2.undistortPoints(right[0][[matches[i] for i in ids], None], c['K2'], c['D2'], P=c['K2'])[:, 0]
    lines = np.column_stack([l, np.ones(len(l))]) @ c['F'].T
    epi = abs((lines * np.column_stack([r, np.ones(len(r))])).sum(1)) / np.linalg.norm(lines[:, :2], axis=1)
    P1 = c['K1'] @ np.column_stack([np.eye(3), np.zeros(3)])
    P2 = c['K2'] @ np.column_stack([c['R'], c['T'].reshape(3)])
    X = cv2.triangulatePoints(P1, P2, l.astype(np.float64).T, r.astype(np.float64).T)
    X = (X[:3] / X[3]).T
    ok = np.isfinite(X).all(1) & (X[:, 2] > 0) & (epi < 1.5)
    return {int(i): x for i, x in zip(ids[ok], X[ok])}


def rigid(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ca).T @ (b-cb))
    R = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    return R, cb - R @ ca


def plane(points):
    ctr = points.mean(0)
    _, _, vt = np.linalg.svd(points-ctr, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    return n, float(n @ ctr)


def analyze(directory, first, second):
    d = Path(directory)
    meta = json.loads((d/'meta.json').read_text(encoding='utf-8'))
    with np.load(meta['calib']) as data:
        c = {k: data[k] for k in data.files}
    fs = [features(d/f'station_{k:02d}_{tag}.png') for k in [first, second] for tag in ['L', 'R']]
    m0, _ = links(fs[0], fs[1])
    m1, _ = links(fs[2], fs[3])
    mt, H01 = links(fs[0], fs[2])
    p0 = stereo(fs[0], fs[1], m0, c)
    p1 = stereo(fs[2], fs[3], m1, c)
    ids = [i for i, j in mt.items() if i in p0 and j in p1]
    if len(ids) < 30:
        raise RuntimeError(f'Only {len(ids)} four-image correspondences')
    a = np.array([p0[i] for i in ids])
    b = np.array([p1[mt[i]] for i in ids])
    rng = np.random.default_rng(7)
    best = np.zeros(len(a), bool)
    for _ in range(150):
        sample = rng.choice(len(a), 8, replace=False)
        R, t = rigid(a[sample], b[sample])
        mask = np.linalg.norm(a @ R.T + t-b, axis=1) < 0.3
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 30:
        raise RuntimeError('Insufficient rigid correspondences')
    R, t = rigid(a[best], b[best])
    residual = np.linalg.norm(a[best] @ R.T + t-b[best], axis=1)
    n, distance = plane(a[best])
    C = -R.T @ t  # 新站相机中心在旧站左相机坐标系中的位移
    normal = float(C @ n)
    tangential = float(np.linalg.norm(C - normal*n))
    normal0, d0 = plane(np.array(list(p0.values())))
    normal1, d1 = plane(np.array(list(p1.values())))
    result = {
        'first_station': first, 'second_station': second,
        'four_image_matches': len(ids), 'rigid_inliers': int(best.sum()),
        'rigid_residual_median_mm': float(np.median(residual)),
        'rigid_residual_p95_mm': float(np.percentile(residual, 95)),
        'camera_displacement_mm_in_first_camera': C.tolist(),
        'camera_rotation_deg': float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))),
        'board_normal_in_first_camera': n.tolist(),
        'optical_axis_to_board_normal_deg': float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
        'normal_travel_mm': normal, 'tangential_travel_mm': tangential,
        'travel_angle_to_board_plane_deg': float(np.degrees(np.arctan2(abs(normal), tangential))),
        'plane_distance_first_mm': d0, 'plane_distance_second_mm': d1,
        'H_first_to_second_raw_pixels': H01.tolist(),
        'note': 'Diagnostic from SIFT triangulation and a local planar fit; not metrology-grade displacement or mechanical alignment certification.'
    }
    out = d/f'motion_diagnostic_{first:02d}_{second:02d}.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--second', type=int, default=1)
    args = parser.parse_args()
    analyze(args.directory, args.first, args.second)
