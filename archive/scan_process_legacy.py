# -*- coding: utf-8 -*-
"""
扫描式 DIC 处理: 基准/变形两轮扫描 -> 逐站位三维位移场 -> 站间刚体
校正 -> 整板拼接位移云图。

用法:
    python scan_process.py captures/scan_def_xxxxxx [--step 32] [--subset 64]
        [--search 60] [--grid-mm 0.5]

输入: def 目录(meta.json 里带 ref_dir)。
流程:
  1) 每站位: 参考对建立(SIFT单应+子区立体) -> 变形对时序/立体子区匹配
     -> 该站相机系三维位移场;
  2) 站间几何: 相邻站位基准扫描重叠区 3D 对应 -> 刚体变换(Kabsch) ->
     链式统一到站位0相机系;
  3) 复位误差校正: 滑台第二轮复位残差表现为每站一个近似平移的伪位移,
     用相邻站位重叠区位移一致性逐站链式扣除(站位0为基准);
  4) 全局试件平面坐标系, 位移投影为 U/V/W, 按 grid-mm 栅格化拼接整板。
输出(def目录/scan_results/): merged.npz, cloud_U/V/W.png, summary.txt
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scandic.dic_engine import DicEngine
from scandic.processing.offline_dic import SubsetMatcher


def load_station(d, k, tag):
    p = os.path.join(d, f"station_{k:02d}_{tag}.png")
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 256.0
    return img


def kabsch(A, B):
    """求刚体 R,t 使 R@A+t ≈ B (A,B: Nx3)。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, cb - R @ ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("def_dir")
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--subset", type=int, default=64)
    ap.add_argument("--search", type=int, default=60)
    ap.add_argument("--grid-mm", type=float, default=0.5)
    args = ap.parse_args()

    dmeta = json.load(open(os.path.join(args.def_dir, "meta.json"),
                           encoding="utf-8"))
    ref_dir = dmeta["ref_dir"]
    calib = dmeta["calib"]
    n_st = len(dmeta["stations"])
    print(f"{n_st} 站位, ref={ref_dir}\ncalib={calib}")

    cnpz = np.load(calib)
    full = (int(cnpz["image_size"][0]), int(cnpz["image_size"][1]))
    eng = DicEngine(calib, out_size=full)
    mat = SubsetMatcher(half=args.subset // 2, search=args.search)
    margin = args.subset // 2 + args.search + 4

    def h_apply(H, x, y):
        den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
                (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)

    # ---------- 每站位处理 ----------
    stations = []   # 每站: dict(P_ref Nx3, D Nx3, xl yl(参考像素), lp_ref)
    for k in range(n_st):
        rl = load_station(ref_dir, k, "L"); rr = load_station(ref_dir, k, "R")
        dl = load_station(args.def_dir, k, "L"); dr = load_station(args.def_dir, k, "R")
        if any(img is None for img in (rl, rr, dl, dr)):
            print(f"站位{k}: 图像缺失, 跳过"); stations.append(None); continue
        l0, r0 = eng.undistort_pair(rl, rr)
        lp0, rp0 = eng._prep(l0), eng._prep(r0)
        H_rl, ni, nm, ptsL, ptsR = eng._bootstrap_H(lp0, rp0)
        Hinv = np.linalg.inv(H_rl)
        rw0 = cv2.warpPerspective(rp0, H_rl, full)
        # 网格
        f32 = l0.astype(np.float32)
        mu = cv2.blur(f32, (31, 31))
        std = np.sqrt(np.maximum(cv2.blur(f32 * f32, (31, 31)) - mu * mu, 0))
        gxs = np.arange(margin, full[0] - margin, args.step)
        gys = np.arange(margin, full[1] - margin, args.step)
        GX, GY = np.meshgrid(gxs, gys)
        pts_ok = std[GY, GX] > eng.texture_thresh
        idxs = np.argwhere(pts_ok)
        # 参考立体
        P_ref, keep, xl_l, yl_l = [], [], [], []
        for i, j in idxs:
            x, y = float(GX[i, j]), float(GY[i, j])
            m = mat.match(lp0, rw0, x, y)
            if m is None:
                continue
            xw, yw = x + m[0], y + m[1]
            xr, yr = h_apply(Hinv, xw, yw)
            if eng._epi_err(np.float32(x), np.float32(y),
                            np.float32(xr), np.float32(yr)) > eng.epi_thresh:
                continue
            keep.append((x, y, xr, yr))
        if len(keep) < 200:
            print(f"站位{k}: 参考有效点过少({len(keep)}), 跳过")
            stations.append(None); continue
        keep = np.array(keep, np.float32)
        P_ref = eng._triangulate(keep[:, 0], keep[:, 1], keep[:, 2], keep[:, 3])
        # 变形态
        ld, rd = eng.undistort_pair(dl, dr)
        lpd, rpd = eng._prep(ld), eng._prep(rd)
        rwd = cv2.warpPerspective(rpd, H_rl, full)
        rows = []
        for n_i, (x, y, _, _) in enumerate(keep):
            mt = mat.match(lp0, lpd, x, y)
            if mt is None:
                continue
            xl2, yl2 = x + mt[0], y + mt[1]
            ms = mat.match(lpd, rwd, xl2, yl2)
            if ms is None:
                continue
            xw, yw = xl2 + ms[0], yl2 + ms[1]
            xr2, yr2 = h_apply(Hinv, xw, yw)
            if eng._epi_err(np.float32(xl2), np.float32(yl2),
                            np.float32(xr2), np.float32(yr2)) > eng.epi_thresh:
                continue
            rows.append((n_i, xl2, yl2, xr2, yr2))
        if len(rows) < 200:
            print(f"站位{k}: 变形态有效点过少({len(rows)}), 跳过")
            stations.append(None); continue
        rows = np.array(rows, np.float32)
        sel = rows[:, 0].astype(int)
        P_def = eng._triangulate(rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])
        st = {"P": P_ref[sel], "D": P_def - P_ref[sel],
              "px": keep[sel, 0], "py": keep[sel, 1], "lp": lp0}
        stations.append(st)
        print(f"站位{k}: 有效点 {len(sel)}, |D|中位 "
              f"{np.median(np.linalg.norm(st['D'], axis=1))*1000:.1f}um")

    valid_idx = [k for k, s in enumerate(stations) if s is not None]
    if len(valid_idx) < 1:
        sys.exit("没有可用站位")

    # ---------- 站间几何(基准扫描): 相邻站位重叠 3D 对应 -> 刚体链 ----------
    Rk = [np.eye(3)]
    tk = [np.zeros(3)]
    for k in range(1, n_st):
        a, b = stations[k - 1], stations[k]
        if a is None or b is None:
            Rk.append(Rk[-1]); tk.append(tk[-1]); continue
        # 在站位k-1参考左图与站位k参考左图之间做子区匹配, 建 3D 对应
        pairsA, pairsB = [], []
        # 用相位相关先估整体位移(初值)
        from scandic.processing.stitch import register_pair, highpass
        s = 0.25
        sa = cv2.resize(a["lp"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sb = cv2.resize(b["lp"], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gdx, gdy, gncc = register_pair(
            highpass(sa.astype(np.float32)), highpass(sb.astype(np.float32)))
        # register_pair 求全景原点偏移；追踪纹理点在下一图的位置须取反。
        gdx /= -s; gdy /= -s
        if gncc < 0.3:
            print(f"  站间 {k-1}->{k}: 重叠配准失败(NCC={gncc:.2f}), 用上一站变换")
            Rk.append(Rk[-1]); tk.append(tk[-1]); continue
        big = SubsetMatcher(half=args.subset // 2, search=48)
        for n_i in range(0, len(a["px"]), 3):   # 稀疏取点足够拟合刚体
            x, y = float(a["px"][n_i]), float(a["py"][n_i])
            xb, yb = x + gdx, y + gdy
            if not (margin < xb < full[0] - margin and margin < yb < full[1] - margin):
                continue
            m = big.match(a["lp"], b["lp"], x, y, init=(gdx, gdy))
            if m is None:
                continue
            # a 中该点的3D已知; b 中位置 -> 找 b 站最近的网格点3D? 直接
            # 用 b 站的像素->3D 需要 b 站该位置的立体解 —— 用 b 站最近有效点近似
            xb2, yb2 = x + m[0], y + m[1]
            d2 = (b["px"] - xb2) ** 2 + (b["py"] - yb2) ** 2
            j = int(np.argmin(d2))
            if d2[j] > (args.step * 1.5) ** 2:
                continue
            pairsA.append(a["P"][n_i])
            pairsB.append(b["P"][j])
        if len(pairsA) < 30:
            print(f"  站间 {k-1}->{k}: 3D对应过少({len(pairsA)})")
            Rk.append(Rk[-1]); tk.append(tk[-1]); continue
        A = np.array(pairsA); B = np.array(pairsB)
        R, t = kabsch(B, A)   # 把站k坐标映到站k-1
        # 链到站0
        Rk.append(Rk[k - 1] @ R)
        tk.append(Rk[k - 1] @ t + tk[k - 1])
        res = np.linalg.norm((A - (R @ B.T).T - t), axis=1)
        print(f"  站间 {k-1}->{k}: {len(A)}对 刚体拟合残差中位 "
              f"{np.median(res):.3f}mm, 平移 {np.linalg.norm(t):.1f}mm")

    # 统一到站0系
    allP, allD, allK = [], [], []
    for k in valid_idx:
        s = stations[k]
        P0 = (Rk[k] @ s["P"].T).T + tk[k]
        D0 = (Rk[k] @ s["D"].T).T
        allP.append(P0); allD.append(D0); allK.append(np.full(len(P0), k))
    allP = np.concatenate(allP); allD = np.concatenate(allD)
    allK = np.concatenate(allK)

    # ---------- 复位误差校正: 重叠区位移一致性(平移链) ----------
    corr = {valid_idx[0]: np.zeros(3)}
    for a_i, b_i in zip(valid_idx[:-1], valid_idx[1:]):
        Pa, Da = allP[allK == a_i], allD[allK == a_i]
        Pb, Db = allP[allK == b_i], allD[allK == b_i]
        # 重叠 = 3D 近邻(1.5倍网格间距内)
        from scipy.spatial import cKDTree
        tree = cKDTree(Pa)
        dist, ji = tree.query(Pb, distance_upper_bound=2.0)
        m = np.isfinite(dist)
        if m.sum() < 30:
            print(f"  复位校正 {a_i}->{b_i}: 重叠点不足, 跳过")
            corr[b_i] = corr[a_i]; continue
        delta = np.median(Db[m] - (Da[ji[m]] + corr[a_i]), axis=0)
        corr[b_i] = -delta + 0  # 使 b 站与 a 站在重叠区一致
        print(f"  复位校正 站{b_i}: ({delta[0]*1000:+.1f}, {delta[1]*1000:+.1f}, "
              f"{delta[2]*1000:+.1f})um")
    for k in valid_idx:
        allD[allK == k] += corr[k]

    # ---------- 试件平面坐标系 + 栅格化拼接 ----------
    ctr = allP.mean(0)
    _, _, vt = np.linalg.svd(allP[::max(1, len(allP)//8000)] - ctr,
                             full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    if n[2] > 0:
        n = -n
    e1 = np.cross([0, 1, 0], n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    S = (allP - ctr) @ e1
    T = (allP - ctr) @ e2
    U = allD @ e1; V = allD @ e2; W = allD @ n

    g = args.grid_mm
    si = ((S - S.min()) / g).astype(int)
    ti = ((T - T.min()) / g).astype(int)
    nw, nh = si.max() + 1, ti.max() + 1
    print(f"整板范围 {S.max()-S.min():.0f} x {T.max()-T.min():.0f} mm, "
          f"栅格 {nw}x{nh} @ {g}mm")
    outd = os.path.join(args.def_dir, "scan_results")
    os.makedirs(outd, exist_ok=True)
    fields = {}
    for name, val in (("U", U), ("V", V), ("W", W)):
        acc = np.zeros((nh, nw), np.float64)
        cnt = np.zeros((nh, nw), np.float64)
        np.add.at(acc, (ti, si), val)
        np.add.at(cnt, (ti, si), 1)
        f = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        fields[name] = f
        lo, hi = np.nanpercentile(f, (2, 98))
        norm = np.nan_to_num(np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1))
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cm[cnt == 0] = 30
        up = max(1, int(1200 / nw))
        cm = cv2.resize(cm, (nw * up, nh * up), interpolation=cv2.INTER_NEAREST)
        cv2.putText(cm, f"{name} {lo*1000:+.1f}~{hi*1000:+.1f} um", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(outd, f"cloud_{name}.png"), np.flipud(cm))
    np.savez_compressed(os.path.join(outd, "merged.npz"),
                        P=allP, D=allD, station=allK,
                        s=S, t=T, u=U, v=V, w=W, grid_mm=g,
                        U=fields["U"], V=fields["V"], W=fields["W"])
    with open(os.path.join(outd, "summary.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"站位: {valid_idx}\n点数: {len(allP)}\n")
        for name, val in (("U", U), ("V", V), ("W", W)):
            fp.write(f"{name}: 中位 {np.median(val)*1000:+.2f}um  "
                     f"P95|{np.percentile(np.abs(val), 95)*1000:.2f}|um\n")
    print(f"完成 -> {outd}")


if __name__ == "__main__":
    main()
