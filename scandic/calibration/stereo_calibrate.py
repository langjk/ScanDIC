# -*- coding: utf-8 -*-
"""
双目立体标定: 从 calib_capture.py 采的图像对算内参+外参+校正映射。

用法:
    python stereo_calibrate.py captures/stereo_calib_xxx --cols 11 --rows 8 --square 15

输出(存到图像目录):
    stereo_calib.npz   K1 D1 K2 D2 R T E F + 校正 R1 R2 P1 P2 Q + 图像尺寸
    calib_report.txt   RMS / 基线 / 各对重投影误差(便于剔除坏对后重跑)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def make_blob_detector(scale=1.0):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 150 * scale * scale
    p.maxArea = 100000 * scale * scale
    p.filterByCircularity = True
    p.minCircularity = 0.6
    p.filterByConvexity = False
    p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)


def _circles_two_stage(img, pattern):
    """圆点板两级检测: 全分辨率直接 findCirclesGrid 在大图+透视下极不稳定,
    先在 1/4 图上定网格排序, 再回全分辨率局部精定位圆心。"""
    s = 0.25
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, centers = cv2.findCirclesGrid(
        small, pattern, flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=make_blob_detector(s))
    if not ok:
        ok, centers = cv2.findCirclesGrid(
            small, pattern,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
            blobDetector=make_blob_detector(s))
    if not ok:
        return None
    coarse = centers.reshape(-1, 2) / s  # 全分辨率坐标(粗)
    # 相邻点距(全分辨率)
    pitch = float(np.median(np.linalg.norm(np.diff(
        coarse.reshape(pattern[1], pattern[0], 2), axis=1), axis=2)))
    # 全分辨率灰度加权质心精定位。不用斑点检测: 空心环形基准点在全
    # 分辨率下会被圆度过滤拒掉; 暗区质心对实心点和圆环同样成立。
    half = 0.42 * pitch
    H, W = img.shape
    fine = np.empty_like(coarse)
    for i, (cx, cy) in enumerate(coarse):
        for _ in range(2):  # 迭代一次使窗口居中
            x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half) + 1)
            y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half) + 1)
            win = img[y0:y1, x0:x1].astype(np.float32)
            if win.size < 400:
                return None
            thr, _ = cv2.threshold(win.astype(np.uint8), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= half * half
            wgt = np.clip(thr - win, 0, None) * disk  # 暗于阈值的部分, 限定圆盘内
            tot = wgt.sum()
            if tot < 50 * 255:
                return None  # 窗口里没有暗区(粗定位飘了/点被遮挡)
            cx = float((wgt * xx).sum() / tot)
            cy = float((wgt * yy).sum() / tot)
        fine[i] = (cx, cy)
    return fine.astype(np.float32)


def find_corners(path, pattern, kind):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    if kind == "circles":
        corners = _circles_two_stage(img, pattern)
        if corners is None:
            return None, img.shape[::-1]
        return corners, img.shape[::-1]
    found, corners = cv2.findChessboardCornersSB(
        img, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        return None, img.shape[::-1]
    return corners.reshape(-1, 2).astype(np.float32), img.shape[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--square", type=float, required=True,
                    help="棋盘格边长/圆点中心距(mm)")
    ap.add_argument("--type", choices=("circles", "chess"), default="circles")
    ap.add_argument("--skip", default="", help="剔除的对编号, 逗号分隔, 如 3,7")
    args = ap.parse_args()
    pattern = (args.cols, args.rows)
    skip = {int(x) for x in args.skip.split(",") if x.strip()}

    lefts = sorted(glob.glob(os.path.join(args.folder, "left_*.png")))
    print(f"{len(lefts)} 对图像")
    if len(lefts) < 8:
        sys.exit("标定对数太少(<8), 请继续采集")

    # 物方角点坐标(mm)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    obj_pts, l_pts, r_pts, used = [], [], [], []
    size = None
    for lf in lefts:
        idx = int(os.path.basename(lf).split("_")[1].split(".")[0])
        if idx in skip:
            print(f"  对{idx:02d}: 手动剔除")
            continue
        rf = os.path.join(args.folder, f"right_{idx:02d}.png")
        cl, size = find_corners(lf, pattern, args.type)
        cr, _ = find_corners(rf, pattern, args.type)
        if cl is None or cr is None:
            print(f"  对{idx:02d}: 角点检测失败, 跳过 (L:{'OK' if cl is not None else '--'} "
                  f"R:{'OK' if cr is not None else '--'})")
            continue
        # 对称圆点阵有 180° 旋转二义性: 左右点序方向不一致时翻转右目
        if args.type == "circles":
            if float((cl[-1] - cl[0]) @ (cr[-1] - cr[0])) < 0:
                cr = cr[::-1].copy()
        obj_pts.append(objp)
        l_pts.append(cl.reshape(-1, 1, 2))
        r_pts.append(cr.reshape(-1, 1, 2))
        used.append(idx)
    print(f"有效 {len(used)} 对: {used}")
    if len(used) < 8:
        sys.exit("有效对数太少")

    # 单目标定
    rms1, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, l_pts, size, None, None)
    rms2, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, r_pts, size, None, None)
    print(f"单目 RMS: 左 {rms1:.3f}px  右 {rms2:.3f}px")

    # ---- 自动质量闸门 ----
    # ① 单视图重投影峰值(抓质心被遮挡/反光带偏的野点)
    # ② 左右相对位姿偏离中位(抓触发滑帧/异步的坏对)
    def _pose(pts, K, D, i):
        _, rv, tv = cv2.solvePnP(obj_pts[i], pts[i], K, D)
        proj, _ = cv2.projectPoints(obj_pts[i], rv, tv, K, D)
        emax = float(np.sqrt(((proj - pts[i]) ** 2).sum(axis=2)).max())
        Rm, _ = cv2.Rodrigues(rv)
        return emax, Rm, tv

    rel_t = []
    peak = []
    for i in range(len(used)):
        eL, RL, tL = _pose(l_pts, K1, D1, i)
        eR, RR, tR = _pose(r_pts, K2, D2, i)
        Rrel = RR @ RL.T
        rel_t.append((tR - Rrel @ tL).ravel())
        peak.append(max(eL, eR))
    rel_t = np.array(rel_t)
    dev = np.linalg.norm(rel_t - np.median(rel_t, axis=0), axis=1)
    dev_thr = max(1.0, 3 * 1.4826 * np.median(np.abs(dev - np.median(dev))))
    good = [i for i in range(len(used))
            if peak[i] < 1.5 and dev[i] < dev_thr]
    for i in range(len(used)):
        if i not in good:
            print(f"  [闸门] 剔除 对{used[i]:02d}: 峰值重投影 {peak[i]:.2f}px, "
                  f"相对位姿偏离 {dev[i]:.2f}mm")
    if len(good) < 8:
        sys.exit("质量闸门后有效对数太少, 请补采")
    obj_pts = [obj_pts[i] for i in good]
    l_pts = [l_pts[i] for i in good]
    r_pts = [r_pts[i] for i in good]
    used = [used[i] for i in good]
    print(f"闸门后 {len(used)} 对: {used}")
    # 用干净数据重新做单目标定
    rms1, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, l_pts, size, None, None)
    rms2, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, r_pts, size, None, None)
    print(f"闸门后单目 RMS: 左 {rms1:.3f}px  右 {rms2:.3f}px")

    # 立体标定(固定内参)
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, l_pts, r_pts, K1, D1, K2, D2, size,
        criteria=crit, flags=flags)
    baseline = float(np.linalg.norm(T))
    ang = np.degrees(np.arccos((np.trace(R) - 1) / 2))
    print(f"立体 RMS: {rms:.3f}px   基线: {baseline:.2f}mm   相机夹角: {ang:.2f}°")

    # 逐对重投影误差(定位坏对)
    per_pair = []
    for i, idx in enumerate(used):
        _, rv, tv = cv2.solvePnP(obj_pts[i], l_pts[i], K1, D1)
        proj, _ = cv2.projectPoints(obj_pts[i], rv, tv, K1, D1)
        e = np.sqrt(((proj - l_pts[i]) ** 2).sum(axis=2)).mean()
        per_pair.append((idx, float(e)))

    # 校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, alpha=0)

    np.savez(os.path.join(args.folder, "stereo_calib.npz"),
             K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=np.array(size), rms=rms, square_mm=args.square)

    with open(os.path.join(args.folder, "calib_report.txt"), "w",
              encoding="utf-8") as fp:
        fp.write(f"图像尺寸: {size}\n有效对: {used}\n")
        fp.write(f"单目RMS: L={rms1:.3f} R={rms2:.3f}\n立体RMS: {rms:.3f}px\n")
        fp.write(f"基线: {baseline:.3f}mm  相机夹角: {ang:.3f}°\n")
        fp.write(f"K1:\n{K1}\nD1: {D1.ravel()}\nK2:\n{K2}\nD2: {D2.ravel()}\n")
        fp.write(f"R:\n{R}\nT: {T.ravel()}\n\n各对左目重投影误差(px):\n")
        for idx, e in per_pair:
            fp.write(f"  对{idx:02d}: {e:.3f}\n")
    print(f"已保存 stereo_calib.npz / calib_report.txt 到 {args.folder}")
    bad = [f"{idx}" for idx, e in per_pair if e > 2 * np.median([x[1] for x in per_pair])]
    if bad:
        print(f"[提示] 这些对误差偏大, 可用 --skip {','.join(bad)} 剔除后重跑")
    if rms > 0.5:
        print("[提示] 立体 RMS > 0.5px, 建议增加姿态多样性或检查标定板平整度")


if __name__ == "__main__":
    main()
