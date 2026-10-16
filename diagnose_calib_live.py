# -*- coding: utf-8 -*-
"""现场诊断 live_dic_gui "标定F残差偏大" 告警的真实病因。

运行前关闭 live_dic_gui / calib_gui / MV Viewer(相机独占)。用法:
    python diagnose_calib_live.py            # 用最新标定, 平均 4 帧
    python diagnose_calib_live.py --navg 6

做三件事, 都不修改标定、不用标定F去筛匹配:
  1. 复现 live_dic 的 calib_epi(实拍散斑 SIFT 内点对标定F的极线残差中位);
  2. 用同一批内点现场拟合一个全新的基础矩阵 F_fit, 报它的残差 ——
     F_fit 残差小而标定F残差大 => 对应关系没问题, 是存的 R,T/位姿不符
     (动过相机或标定外参不准); 两者都大 => 匹配/畸变问题;
  3. 把标定F残差按到像主点的半径分三档 => 残差随半径明显变大 => 畸变过拟合。
顺便把这对 16bit 原始图存到 captures/calibdiag_*/, 供 offline 工具复核。
"""
import argparse
import os
import time

import cv2
import numpy as np

from calib_gui import CamThread, TriggerPump, LEFT_SN, RIGHT_SN
from dic_engine import DicEngine
from live_dic_gui import DEFAULT_CALIB, ROOT


def grab_average(thL, thR, n_avg, timeout=20):
    """同步对拍取 n_avg 个不同帧求平均(软触发已在 TriggerPump 里跑)。"""
    accs = [None, None]
    cnts = [0, 0]
    prev = [None, None]
    t0 = time.time()
    while min(cnts) < n_avg and time.time() - t0 < timeout:
        for i, th in enumerate((thL, thR)):
            f = th.latest()
            if f is None or f is prev[i] or cnts[i] >= n_avg:
                continue
            prev[i] = f
            accs[i] = f.astype(np.float32) if accs[i] is None else accs[i] + f
            cnts[i] += 1
        time.sleep(0.01)
    if accs[0] is None or accs[1] is None:
        raise RuntimeError("取帧失败, 检查相机是否被占用")
    return accs[0] / cnts[0], accs[1] / cnts[1], cnts


def radial_bins(pts, residual, cx, cy, w):
    """按到像主点半径三等分, 报各档残差中位(全分辨率等效), 看是否随半径增大。"""
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    edges = np.quantile(r, [0, 1 / 3, 2 / 3, 1.0])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r <= hi)
        if m.sum() >= 5:
            out.append((lo, hi, m.sum(), float(np.median(np.abs(residual[m])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navg", type=int, default=4)
    args = ap.parse_args()

    cal = DEFAULT_CALIB
    print(f"[标定] {cal}")
    eng = DicEngine(cal)                      # 默认 out_size=(1024,768), 与 live 一致
    s = eng.out_size[0] / 4096.0              # 全分辨率->缩放域比例(仅用于换算打印)

    thL, thR = CamThread(LEFT_SN), CamThread(RIGHT_SN)
    thL.error.connect(lambda sn, m: print(f"[{sn}] {m}"))
    thR.error.connect(lambda sn, m: print(f"[{sn}] {m}"))
    thL.start(); thR.start()
    pump = TriggerPump(thL, thR); pump.start()
    try:
        print(f"预热取帧(平均 {args.navg} 帧)...")
        time.sleep(1.5)
        rawL, rawR, cnts = grab_average(thL, thR, args.navg)
        print(f"取到 L/R = {cnts} 帧")

        # 存盘原始对(16bit, 与 SeqRecorder 同格式)
        out = os.path.join(ROOT, "captures", time.strftime("calibdiag_%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        for img, tag in ((rawL, "L"), (rawR, "R")):
            u16 = np.clip(img * 256.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(out, f"pair_{tag}.png"), u16,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # 复现 live_dic 的参考建立(含 calib_epi)
        eng._ref_acc = None
        eng.accumulate_reference(rawL, rawR)
        diag = eng.finish_reference()
        ce = diag["calib_epi"]
        print("\n===== 诊断结果(残差单位: 缩放域px / 全分辨率等效px) =====")
        print(f"calib_epi 标定F残差 = {ce:.3f}  (全分辨率≈{ce/s:.2f}px)   "
              f"[live 告警阈值 0.8]")
        print(f"epi_med   光流残差   = {diag['epi_med']:.3f}")
        print(f"有效区 {diag['valid_frac']*100:.0f}%   "
              f"SIFT 内点 {diag['sift_inliers']}/{diag['sift_matches']}   "
              f"Z {diag['z_range'][0]:.0f}~{diag['z_range'][1]:.0f}mm")

        # 重取内点(finish_reference 没外传), 直接复算以拟合 F_fit
        l = np.clip(eng.ref["l"], 0, 255).astype(np.uint8)
        # 用已存的参考预处理图重跑自举拿内点坐标
        lp = eng.ref["lp"]
        r_u = cv2.remap(rawR, *eng.mapR, cv2.INTER_LINEAR)
        rp = eng._prep(r_u)
        _, _, _, ptsL, ptsR = eng._bootstrap_H(lp, rp)
        res_cal = eng._epi_err(ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1])

        F_fit, inl = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 0.5, 0.999)
        if F_fit is not None:
            lines = np.column_stack([ptsL, np.ones(len(ptsL))]) @ F_fit.T
            res_fit = np.abs((lines * np.column_stack(
                [ptsR, np.ones(len(ptsR))])).sum(1)) / np.linalg.norm(lines[:, :2], axis=1)
            mfit = float(np.median(res_fit))
            print(f"\nF_fit 现场重拟合残差 = {mfit:.3f}  (全分辨率≈{mfit/s:.2f}px)   "
                  f"内点 {int(inl.sum())}/{len(ptsL)}")
            print(f"  -> 标定F/现拟合 残差比 = {ce/max(mfit,1e-6):.1f}x")
            if ce > 0.8 and mfit < 0.3:
                print("  判读: 对应关系良好但标定外参不符 => 相机动过 或 标定R,T不准")
            elif ce > 0.8 and mfit >= 0.3:
                print("  判读: 现拟合也大 => 匹配/畸变问题(反光/失焦/畸变过拟合)")

        print("\n标定F残差按半径分档(到像主点):")
        cx, cy = eng.K1[0, 2], eng.K1[1, 2]
        for lo, hi, n, med in radial_bins(ptsL, res_cal, cx, cy, eng.out_size[0]):
            print(f"  r {lo:6.0f}~{hi:6.0f}px  n={n:4d}  残差中位 {med:.3f}  "
                  f"(全分辨率≈{med/s:.2f}px)")
        print("  -> 边档远大于中心档 => 畸变外推(k3 过拟合)是主因之一")

        print(f"\n原始图对已存: {out}")
    finally:
        pump.stop(); thL.stop(); thR.stop()


if __name__ == "__main__":
    main()
