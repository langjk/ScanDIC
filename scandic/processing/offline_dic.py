# -*- coding: utf-8 -*-
"""
离线高精度三维 DIC: 处理 live_dic_gui 序列采集(seq_xxx)的多帧平均图像对。

与实时模式的区别(精度需通过已知位移实验评估):
  - 全分辨率(4096x3000)处理, 实时为速度用 1/4 分辨率;
  - 子区匹配(整像素 ZNCC + 亚像素相位相关)替代稠密光流;
  - 每状态多帧平均图像, 传感器噪声已除 sqrt(N)。

用法:
    python offline_dic.py captures/seq_xxxxxx [--step 32] [--subset 64]
        [--search 40] [--ref 0]

输出(存到 seq_dir/results/):
    state_XXXX.npz   grid_x/grid_y(px), u/v/w(mm, 试件系), valid
    summary.csv      每状态 u/v/w 的中位/均值/P95，单位 um
    cloud_XXXX.png   W 分量云图
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scandic.dic_engine import DicEngine
from scandic.processing.stitch import phase_correlate_even


def load_state(d, idx, tag):
    p = os.path.join(d, f"state_{idx:04d}_{tag}.png")
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 256.0
    return img


class SubsetMatcher:
    def __init__(self, half=32, search=40, min_ncc=0.3):
        if half < 2 or search < 0:
            raise ValueError("half 必须 >=2，search 必须 >=0")
        self.half = half
        self.search = search
        self.min_ncc = min_ncc
        self._hann = cv2.createHanningWindow((2 * half, 2 * half), cv2.CV_32F)

    def match(self, a, b, x, y, init=(0.0, 0.0)):
        """a 中 (x,y) 处子区在 b 中的位移 (dx, dy, ncc); 失败返回 None。"""
        h, w = a.shape
        bh, bw = b.shape
        half, sr = self.half, self.search
        tx0, ty0 = int(round(x)) - half, int(round(y)) - half
        if tx0 < 0 or ty0 < 0 or tx0 + 2 * half > w or ty0 + 2 * half > h:
            return None
        tmpl = a[ty0:ty0 + 2 * half, tx0:tx0 + 2 * half].astype(np.float32)
        if float(tmpl.std()) < 1e-6:
            return None
        cx, cy = int(round(x + init[0])), int(round(y + init[1]))
        sx0, sy0 = cx - half - sr, cy - half - sr
        sx1, sy1 = cx + half + sr, cy + half + sr
        if sx0 < 0 or sy0 < 0 or sx1 > bw or sy1 > bh:
            return None
        res = cv2.matchTemplate(b[sy0:sy1, sx0:sx1].astype(np.float32), tmpl,
                                cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if not np.isfinite(mv) or mv < self.min_ncc:
            return None
        bx, by = sx0 + ml[0], sy0 + ml[1]
        bw = b[by:by + 2 * half, bx:bx + 2 * half]
        aw = tmpl.astype(np.float32).copy()
        bwf = bw.astype(np.float32).copy()
        (sxp, syp), _ = phase_correlate_even(aw, bwf, self._hann)
        if not np.isfinite([sxp, syp]).all() or abs(sxp) > 3 or abs(syp) > 3:
            sxp = syp = 0.0  # 亚像素失败, 退回整像素
        # 这里求 a 中纹理点在 b 中的运动，与 stitch 的全景原点位移方向相反。
        # phaseCorrelate(a,b) 返回纹理的正向残差，应加回整像素搜索位置。
        return (bx - tx0) + sxp, (by - ty0) + syp, mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("--step", type=int, default=32, help="网格步长 px")
    ap.add_argument("--subset", type=int, default=64, help="子区尺寸 px")
    ap.add_argument("--search", type=int, default=40, help="搜索半径 px")
    ap.add_argument("--ref", type=int, default=0, help="参考状态编号")
    ap.add_argument("--output", default=None, help="结果目录；可另存以保留历史结果")
    args = ap.parse_args()
    d = args.seq_dir

    with open(os.path.join(d, "meta.json"), encoding="utf-8") as fp:
        meta = json.load(fp)
    calib = meta["calib"]
    n_states = len(glob.glob(os.path.join(d, "state_*_L.png")))
    print(f"{n_states} 个状态, 标定: {calib}")

    cnpz = np.load(calib)
    full = (int(cnpz["image_size"][0]), int(cnpz["image_size"][1]))
    eng = DicEngine(calib, out_size=full)   # 全分辨率
    mat = SubsetMatcher(half=args.subset // 2, search=args.search)

    # ---- 参考 ----
    refL = load_state(d, args.ref, "L")
    refR = load_state(d, args.ref, "R")
    if refL is None or refR is None:
        sys.exit(f"读不到参考状态 {args.ref}")
    l0, r0 = eng.undistort_pair(refL, refR)
    lp0, rp0 = eng._prep(l0), eng._prep(r0)
    eng.H_rl, ni, nm, ptsL, ptsR = eng._bootstrap_H(lp0, rp0)
    cal_epi = float(np.median(eng._epi_err(ptsL[:, 0], ptsL[:, 1],
                                           ptsR[:, 0], ptsR[:, 1])))
    print(f"SIFT {nm}/{ni}, 标定F残差 {cal_epi:.2f}px"
          + ("  [警告: 偏大, 建议重标!]" if cal_epi > 0.8 else ""))
    rw0 = cv2.warpPerspective(rp0, eng.H_rl, full)
    Hinv = np.linalg.inv(eng.H_rl)

    # 网格(纹理达标 + 边距)
    f32 = l0.astype(np.float32)
    mu = cv2.blur(f32, (31, 31))
    std = np.sqrt(np.maximum(cv2.blur(f32 * f32, (31, 31)) - mu * mu, 0))
    margin = args.subset // 2 + args.search + 4
    gx = np.arange(margin, full[0] - margin, args.step)
    gy = np.arange(margin, full[1] - margin, args.step)
    GX, GY = np.meshgrid(gx, gy)
    ny, nx = GX.shape
    tex_ok = std[GY, GX] > eng.texture_thresh
    print(f"网格 {nx}x{ny} = {nx*ny}, 纹理达标 {tex_ok.mean()*100:.0f}%")

    def h_apply(H, x, y):
        den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
                (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)

    def stereo_at(lp_img, rw_img, xs, ys, valid, init_dw):
        """左图(xs,ys)处与 H 扭正右图匹配 -> 原右图坐标; init_dw 为扭正域初值。"""
        xr = np.full(xs.shape, np.nan, np.float32)
        yr = np.full(xs.shape, np.nan, np.float32)
        dw = np.zeros(xs.shape + (2,), np.float32)
        for idx in np.argwhere(valid):
            i, j = idx
            m = mat.match(lp_img, rw_img, xs[i, j], ys[i, j],
                          init=tuple(init_dw[i, j]))
            if m is None:
                continue
            dxw, dyw, _ = m
            dw[i, j] = (dxw, dyw)
            xw, yw = xs[i, j] + dxw, ys[i, j] + dyw
            xr[i, j], yr[i, j] = h_apply(Hinv, xw, yw)
        return xr, yr, dw

    print("参考立体匹配...")
    zero = np.zeros(GX.shape + (2,), np.float32)
    xr0, yr0, dw0 = stereo_at(lp0, rw0, GX.astype(np.float32),
                              GY.astype(np.float32), tex_ok, zero)
    ok0 = tex_ok & np.isfinite(xr0)
    epi0 = eng._epi_err(GX.astype(np.float32), GY.astype(np.float32), xr0, yr0)
    ok0 &= epi0 < eng.epi_thresh
    X0, Y0, Z0 = [np.full(GX.shape, np.nan, np.float32) for _ in range(3)]
    P0 = eng._triangulate(GX[ok0].astype(np.float32),
                          GY[ok0].astype(np.float32), xr0[ok0], yr0[ok0])
    X0[ok0], Y0[ok0], Z0[ok0] = P0[:, 0], P0[:, 1], P0[:, 2]
    print(f"参考有效点 {ok0.sum()}/{nx*ny}, 极线中位 "
          f"{np.median(epi0[ok0]):.3f}px, Z {np.nanmedian(Z0):.0f}mm")

    # 试件平面坐标系
    pts = P0
    ctr = pts.mean(0)
    _, _, vt = np.linalg.svd(pts[::max(1, len(pts)//5000)] - ctr,
                             full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    if n[2] > 0:
        n = -n
    e1 = np.cross([0, 1, 0], n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    basis = np.stack([e1, e2, n])

    outd = args.output or os.path.join(d, "results")
    os.makedirs(outd, exist_ok=True)
    summ = open(os.path.join(outd, "summary.csv"), "w", encoding="utf-8",
                newline="")
    sw = csv.writer(summ)
    sw.writerow(["state", "n_valid", "u_med", "v_med", "w_med",
                 "u_mean", "v_mean", "w_mean", "u_p95", "v_p95", "w_p95"])

    prev_t = np.zeros(GX.shape + (2,), np.float32)  # 时序初值(增量跟踪)
    for k in range(n_states):
        if k == args.ref:
            continue
        curL, curR = load_state(d, k, "L"), load_state(d, k, "R")
        if curL is None or curR is None:
            continue
        l, r = eng.undistort_pair(curL, curR)
        lp, rp = eng._prep(l), eng._prep(r)
        rw = cv2.warpPerspective(rp, eng.H_rl, full)
        # 时序子区匹配(参考->当前), 初值用上一状态
        xl = np.full(GX.shape, np.nan, np.float32)
        yl = np.full(GX.shape, np.nan, np.float32)
        for idx in np.argwhere(ok0):
            i, j = idx
            m = mat.match(lp0, lp, GX[i, j], GY[i, j], init=tuple(prev_t[i, j]))
            if m is None:
                continue
            xl[i, j] = GX[i, j] + m[0]
            yl[i, j] = GY[i, j] + m[1]
            prev_t[i, j] = (m[0], m[1])
        okt = ok0 & np.isfinite(xl)
        # 当前立体(初值用参考的扭正域位移)
        xr, yr, _ = stereo_at(lp, rw, xl, yl, okt, dw0)
        okc = okt & np.isfinite(xr)
        epi = eng._epi_err(xl, yl, xr, yr)
        okc &= epi < eng.epi_thresh
        u = np.full(GX.shape, np.nan, np.float32)
        v = np.full(GX.shape, np.nan, np.float32)
        w = np.full(GX.shape, np.nan, np.float32)
        Pc = eng._triangulate(xl[okc], yl[okc], xr[okc], yr[okc])
        D = Pc - np.stack([X0[okc], Y0[okc], Z0[okc]], axis=-1)
        u[okc], v[okc], w[okc] = D @ basis[0], D @ basis[1], D @ basis[2]
        np.savez_compressed(
            os.path.join(outd, f"state_{k:04d}.npz"),
            grid_x=GX, grid_y=GY, u=u, v=v, w=w, valid=okc)
        vals = [u[okc], v[okc], w[okc]]
        sw.writerow([k, int(okc.sum())]
                    + [f"{np.median(x)*1000:.2f}" for x in vals]
                    + [f"{x.mean()*1000:.2f}" for x in vals]
                    + [f"{np.percentile(np.abs(x), 95)*1000:.2f}" for x in vals])
        summ.flush()
        # W 云图
        wf = np.where(okc, w, np.nan)
        lo, hi = np.nanpercentile(wf, (2, 98))
        norm = np.clip((wf - lo) / max(hi - lo, 1e-9), 0, 1)
        norm = np.nan_to_num(norm)
        small = cv2.resize(norm, (nx * 8, ny * 8), interpolation=cv2.INTER_LINEAR)
        cm = cv2.applyColorMap((small * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        mres = cv2.resize(okc.astype(np.uint8), (nx * 8, ny * 8),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        gsm = cv2.resize(l, (nx * 8, ny * 8), interpolation=cv2.INTER_AREA)
        gsm = np.clip(gsm, 0, 255).astype(np.uint8)
        g3 = cv2.cvtColor(gsm, cv2.COLOR_GRAY2BGR)
        vis = np.where(mres[..., None], (0.65 * cm + 0.35 * g3).astype(np.uint8), g3)
        cv2.putText(vis, f"W {lo*1000:+.1f}~{hi*1000:+.1f} um", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(outd, f"cloud_{k:04d}.png"), vis)
        print(f"状态 {k}: 有效 {okc.sum()}  "
              f"u/v/w 中位 {np.median(u[okc])*1000:+.2f}/"
              f"{np.median(v[okc])*1000:+.2f}/{np.median(w[okc])*1000:+.2f} um")
    summ.close()
    print(f"完成 -> {outd}")


if __name__ == "__main__":
    main()
