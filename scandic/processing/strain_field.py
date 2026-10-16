# -*- coding: utf-8 -*-
"""逐站曲面 Green-Lagrange 应变 —— 对相机刚体运动精确免疫。

原理: 每站在其自身相机系里, 对每个测点用规距 L 内的邻点拟合变形梯度
G(3x2, 参考切平面 -> 变形态 3D), 取 E = 1/2 (G^T G - I2)。
若该站相机有任意刚体位姿误差 (R,t): x -> R x + t  =>  G -> R G,
而 G^T G -> G^T R^T R G = G^T G 精确不变 => E 不变。
故应变不需要固定基准/锚点/站间规范, 只能且必须在"拼接之前、逐站"算。

主应变/最大剪/等效应变是基无关不变量, 可跨站直接比较与拼接。

输入: precision_scan_pipeline 产出的 precision_results/station_XX.npz
  P (N,3) 参考态 3D(相机系, mm), D (N,3) 位移(相机系, mm)。
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def plane_basis(P):
    """SVD 拟合点云切平面, 返回行向量基 (e1,e2,n)。"""
    c = P.mean(0)
    Q = P - c
    # 抽样加速, 对大点云足够稳
    s = Q[:: max(1, len(Q) // 5000)]
    _, _, vt = np.linalg.svd(s, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    if n[2] > 0:
        n = -n
    e1 = np.cross([0.0, 1.0, 0.0], n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n]), c


def green_lagrange_strain(P, D, gauge_mm=8.0, min_neighbors=8, basis=None):
    """逐点曲面 Green-Lagrange 应变。

    返回 dict:
      E     (N,3)  [Exx, Eyy, Exy] 在站内切平面基(或传入 basis)下, 无效点 NaN
      e1,e2 (N,)   主应变 (e1>=e2)
      gamma_max (N,) 最大剪 = e1 - e2
      eq    (N,)   等效(von Mises 面内)应变 sqrt(e1^2 - e1 e2 + e2^2)
      valid (N,) bool
      basis (3,3)
    高斯加权最小二乘拟合 G; sigma = gauge/2。
    """
    P = np.asarray(P, float)
    D = np.asarray(D, float)
    N = len(P)
    if basis is None:
        basis, _ = plane_basis(P)
    e1v, e2v = basis[0], basis[1]
    X = P                      # 参考态 3D
    x = P + D                  # 变形态 3D
    xi = np.stack([(X - X.mean(0)) @ e1v, (X - X.mean(0)) @ e2v], axis=1)  # 仅用于 KDTree 的平面坐标
    tree = cKDTree(xi)
    sigma = gauge_mm / 2.0
    I2 = np.eye(2)

    E = np.full((N, 3), np.nan)
    pe1 = np.full(N, np.nan)
    pe2 = np.full(N, np.nan)
    gmax = np.full(N, np.nan)
    eq = np.full(N, np.nan)
    valid = np.zeros(N, bool)

    for i in range(N):
        idx = tree.query_ball_point(xi[i], gauge_mm)
        if len(idx) < min_neighbors:
            continue
        idx = np.asarray(idx)
        # 参考切平面局部坐标(相对中心点 i)
        dXi = (X[idx] - X[i])
        xi_loc = np.stack([dXi @ e1v, dXi @ e2v], axis=1)       # (m,2)
        dx = (x[idx] - x[i])                                     # (m,3) 变形态位移差
        d2 = (xi_loc ** 2).sum(1)
        w = np.exp(-d2 / (2 * sigma * sigma))                   # 高斯权
        # 加权正规方程: G = (Σ w dx xiᵀ)(Σ w xi xiᵀ)^-1   G:3x2
        A = (w[:, None, None] * xi_loc[:, None, :] * xi_loc[:, :, None]).sum(0)  # 2x2
        if np.linalg.cond(A) > 1e8:
            continue
        B = (w[:, None, None] * dx[:, :, None] * xi_loc[:, None, :]).sum(0)      # 3x2
        G = B @ np.linalg.inv(A)                                 # 3x2
        Emat = 0.5 * (G.T @ G - I2)                              # 2x2 对称
        exx, eyy, exy = Emat[0, 0], Emat[1, 1], Emat[0, 1]
        E[i] = [exx, eyy, exy]
        tr = exx + eyy
        dsc = np.sqrt(max(((exx - eyy) / 2) ** 2 + exy * exy, 0.0))
        a, b = tr / 2 + dsc, tr / 2 - dsc
        pe1[i], pe2[i] = a, b
        gmax[i] = a - b
        eq[i] = np.sqrt(max(a * a - a * b + b * b, 0.0))
        valid[i] = True
    return dict(E=E, e1=pe1, e2=pe2, gamma_max=gmax, eq=eq, valid=valid, basis=basis)


def _st(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    a = np.abs(v)
    return dict(n=int(len(v)), median=float(np.median(v)),
                abs_median=float(np.median(a)),
                abs_p95=float(np.percentile(a, 95)),
                rms=float(np.sqrt(np.mean(v * v))))


def process_dir(precision_dir, gauge_mm=8.0, min_neighbors=8):
    files = sorted(glob.glob(os.path.join(precision_dir, "station_[0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"{precision_dir} 下没有 station_XX.npz")
    per_station = []
    all_eq, all_gmax, all_exx, all_eyy, all_exy = [], [], [], [], []
    all_dmag = []
    for f in files:
        d = np.load(f)
        P, D = d["P"], d["D"]
        r = green_lagrange_strain(P, D, gauge_mm, min_neighbors)
        v = r["valid"]
        dmag = np.linalg.norm(D, axis=1) * 1e3  # um
        # 转微应变 ue
        rec = dict(
            station=os.path.basename(f),
            points=int(len(P)), valid=int(v.sum()),
            disp_abs_median_um=float(np.median(np.abs(dmag))),
            disp_abs_p95_um=float(np.percentile(np.abs(dmag), 95)),
            Exx_ue=_st(r["E"][v, 0] * 1e6), Eyy_ue=_st(r["E"][v, 1] * 1e6),
            Exy_ue=_st(r["E"][v, 2] * 1e6),
            eq_ue=_st(r["eq"][v] * 1e6), gamma_max_ue=_st(r["gamma_max"][v] * 1e6))
        per_station.append(rec)
        all_exx.append(r["E"][v, 0]); all_eyy.append(r["E"][v, 1]); all_exy.append(r["E"][v, 2])
        all_eq.append(r["eq"][v]); all_gmax.append(r["gamma_max"][v]); all_dmag.append(dmag)
    cat = lambda L: np.concatenate(L) if L else np.array([])
    overall = dict(
        gauge_mm=gauge_mm, min_neighbors=min_neighbors,
        stations=len(files),
        disp_abs_p95_um=float(np.percentile(np.abs(cat(all_dmag)), 95)),
        Exx_ue=_st(cat(all_exx) * 1e6), Eyy_ue=_st(cat(all_eyy) * 1e6),
        Exy_ue=_st(cat(all_exy) * 1e6),
        eq_ue=_st(cat(all_eq) * 1e6), gamma_max_ue=_st(cat(all_gmax) * 1e6))
    return dict(overall=overall, per_station=per_station)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("precision_dir", help="含 station_XX.npz 的目录")
    ap.add_argument("--gauge-mm", type=float, default=8.0)
    ap.add_argument("--min-neighbors", type=int, default=8)
    ap.add_argument("--sweep", action="store_true", help="规距扫描 4/6/8/12mm")
    args = ap.parse_args()
    if args.sweep:
        for L in (4.0, 6.0, 8.0, 12.0):
            r = process_dir(args.precision_dir, L, args.min_neighbors)
            o = r["overall"]
            print(f"L={L:4.1f}mm  eq应变(ue) 中位{o['eq_ue']['median']:7.1f} "
                  f"P95{o['eq_ue']['abs_p95']:8.1f}  RMS{o['eq_ue']['rms']:8.1f}   "
                  f"Exx_RMS{o['Exx_ue']['rms']:7.1f} Eyy_RMS{o['Eyy_ue']['rms']:7.1f} "
                  f"Exy_RMS{o['Exy_ue']['rms']:7.1f}  (位移P95 {o['disp_abs_p95_um']:.1f}um)")
        return
    r = process_dir(args.precision_dir, args.gauge_mm, args.min_neighbors)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
