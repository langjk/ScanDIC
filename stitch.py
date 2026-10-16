# -*- coding: utf-8 -*-
"""
散斑扫描图像拼接

把滑台扫描采集的一串帧(相邻帧有重叠)拼成整板全景图。
配准 = 粗配(降采样模板匹配) + 精配(重叠区相位相关, 亚像素),
散斑纹理相关峰非常尖锐, 平移估计可达 0.1px 量级。

用法:
    python stitch.py captures/run_xxxx -o panorama.png
    python stitch.py captures/run_xxxx -o pano.png --flat --min-shift 2

输出:
    panorama.png     拼接结果
    positions.csv    每帧在全景中的位置(x, y, 亚像素), 供后续 DIC 使用
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np


# ---------------- 配准 ----------------

def highpass(gray_f32):
    """去低频(暗角/照明梯度): 降采样-模糊-放大作低频估计, 相减。

    配准必须在高通图上做: 低频梯度会给相位相关的亚像素估计带来
    ~0.2px/对的系统性偏置, 逐对累积。
    """
    h, w = gray_f32.shape
    small = cv2.resize(gray_f32, (max(8, w // 8), max(8, h // 8)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    low = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return gray_f32 - low


_HANN_CACHE = {}


def _hann(shape):
    key = (shape[1], shape[0])
    if key not in _HANN_CACHE:
        _HANN_CACHE[key] = cv2.createHanningWindow(key, cv2.CV_32F)
    return _HANN_CACHE[key]


def phase_correlate_even(a, b, window):
    """保护输入并保证实际 DFT 尺寸为偶数。

    仅裁成偶数不够，例如122会被OpenCV内部补到125，仍产生0.5px偏置。
    先在原子区加窗，再补零到偶数最优DFT尺寸，不裁掉测量数据。
    """
    aw = a.astype(np.float32) * window
    bw = b.astype(np.float32) * window
    h, w = aw.shape
    dh, dw = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    while dh % 2:
        dh = cv2.getOptimalDFTSize(dh + 1)
    while dw % 2:
        dw = cv2.getOptimalDFTSize(dw + 1)
    if (dh, dw) != (h, w):
        aw = cv2.copyMakeBorder(aw, 0, dh-h, 0, dw-w, cv2.BORDER_CONSTANT)
        bw = cv2.copyMakeBorder(bw, 0, dh-h, 0, dw-w, cv2.BORDER_CONSTANT)
    return cv2.phaseCorrelate(aw, bw)


def _overlap_ncc(prev_gray, cur_gray, dx, dy):
    """按整数位移取重叠区, 算归一化互相关作为配准质量。"""
    h, w = prev_gray.shape
    idx, idy = int(round(dx)), int(round(dy))
    x0, x1 = max(0, idx), min(w, w + idx)
    y0, y1 = max(0, idy), min(h, h + idy)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return 0.0
    a = prev_gray[y0:y1, x0:x1]
    b = cur_gray[y0 - idy : y1 - idy, x0 - idx : x1 - idx]
    if x1 - x0 > 400 and y1 - y0 > 200:  # 大重叠区降采样提速, 散斑对齐时相关性保留
        a = cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _refine_subpixel(prev_gray, cur_gray, dx, dy):
    """在整数对齐的重叠区上做相位相关, 得到亚像素残差修正。"""
    h, w = prev_gray.shape
    idx, idy = int(round(dx)), int(round(dy))
    x0, x1 = max(0, idx), min(w, w + idx)
    y0, y1 = max(0, idy), min(h, h + idy)
    # 尺寸必须为偶数: OpenCV 5.0 phaseCorrelate 在奇数尺寸下峰值反卷绕
    # 有恰好 0.5px 的系统偏移(实测 675 行时 sy 偏 +0.5)
    x1 -= (x1 - x0) & 1
    y1 -= (y1 - y0) & 1
    if x1 - x0 > 64 and y1 - y0 > 64:
        # 必须拷贝: phaseCorrelate 在无需 DFT 补边时会原地加窗, 污染原图
        a = prev_gray[y0:y1, x0:x1].astype(np.float32).copy()
        b = cur_gray[y0 - idy : y1 - idy, x0 - idx : x1 - idx].astype(np.float32).copy()
        (sx, sy), _ = phase_correlate_even(a, b, _hann(a.shape))
        # phaseCorrelate(a,b) 返回值取负 = b 相对 a 的位置残差
        if abs(sx) < 8 and abs(sy) < 8:
            return idx - sx, idy - sy
    return float(idx), float(idy)


def _coarse_template(prev_gray, cur_gray, ds=0.5):
    """回退粗配: 从 cur 取左/中/右三个小模板在 prev 中匹配, 取峰值最高者。

    模板小(1/5 幅宽), 三个横向位置保证大位移时至少一个模板落在重叠区内。
    """
    p = cv2.resize(prev_gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    c = cv2.resize(cur_gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    ch, cw = c.shape
    th, tw = int(ch * 0.5), int(cw * 0.2)
    ty0 = (ch - th) // 2
    best = (-1.0, 0.0, 0.0)
    for fx in (0.08, 0.4, 0.72):
        tx0 = int(cw * fx)
        tmpl = c[ty0 : ty0 + th, tx0 : tx0 + tw]
        res = cv2.matchTemplate(p, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best[0]:
            best = (mv, (ml[0] - tx0) / ds, (ml[1] - ty0) / ds)
    return best[1], best[2]


def register_pair(prev_gray, cur_gray):
    """估计 cur 相对 prev 的平移(cur 原点在 prev 坐标系中的位置)。

    粗配: 全帧相位相关(位移在半幅以内均可测, 散斑相关峰极尖);
    失败时回退多模板匹配。精配: 重叠区相位相关(亚像素)。
    返回 (dx, dy, ncc)。
    """
    # 尺寸取偶(奇数尺寸 phaseCorrelate 有 0.5px 偏移 bug);
    # 必须拷贝: phaseCorrelate 在无需 DFT 补边时会原地加窗, 污染原图
    he = prev_gray.shape[0] & ~1
    we = prev_gray.shape[1] & ~1
    a = prev_gray[:he, :we].astype(np.float32).copy()
    b = cur_gray[:he, :we].astype(np.float32).copy()
    (sx, sy), _ = phase_correlate_even(a, b, _hann(a.shape))
    dx, dy = -sx, -sy
    ncc = _overlap_ncc(prev_gray, cur_gray, dx, dy)
    if ncc < 0.4:  # 相位相关失败(位移过半幅/纹理弱), 回退模板匹配
        dx2, dy2 = _coarse_template(prev_gray, cur_gray)
        ncc2 = _overlap_ncc(prev_gray, cur_gray, dx2, dy2)
        if ncc2 > ncc:
            dx, dy, ncc = dx2, dy2, ncc2
    dx, dy = _refine_subpixel(prev_gray, cur_gray, dx, dy)
    ncc = _overlap_ncc(prev_gray, cur_gray, dx, dy)
    return dx, dy, ncc


# ---------------- 光照平场 ----------------

def flat_field(frames_gray):
    """用所有帧的平均近似照明分布(散斑随机移动, 平均后纹理被抹平)。"""
    acc = np.zeros_like(frames_gray[0], dtype=np.float64)
    for f in frames_gray:
        acc += f
    avg = acc / len(frames_gray)
    avg = cv2.GaussianBlur(avg.astype(np.float32), (0, 0), 31)
    avg /= avg.mean()
    np.clip(avg, 0.3, 3.0, out=avg)
    return avg  # 每帧除以它即校正暗角/照明不均


# ---------------- 融合 ----------------

def feather_weight(h, w, feather):
    """边缘线性衰减的权重图(平顶), 用于实时预览的粗融合。"""
    ramp_x = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float32)
    ramp_y = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(np.float32)
    ramp_x = np.clip(ramp_x / feather, 0, 1)
    ramp_y = np.clip(ramp_y / feather, 0, 1)
    return np.outer(ramp_y, ramp_x)


def blend_weight(h, w, power=4):
    """中心占优权重(帐篷函数的高次幂), 用于最终融合。

    宽重叠区若做接近 50/50 的平均, 帧间 1~2px 残余错位会把散斑边缘
    抹虚(实测锐度掉一半); 高次幂让每个像素基本只取帧中心离它最近的
    那一帧, 过渡带收窄, 保持源图锐度。
    """
    tx = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float32)
    ty = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(np.float32)
    tx /= tx.max()
    ty /= ty.max()
    return np.outer(ty, tx) ** power


def composite(frames, positions, power=4):
    """按亚像素位置将各帧以中心占优权重融合到画布上。frames: HxWx3 或 HxW。"""
    h, w = frames[0].shape[:2]
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    x_min, y_min = np.floor(xs.min()), np.floor(ys.min())
    W = int(np.ceil(xs.max() + w) - x_min)
    H = int(np.ceil(ys.max() + h) - y_min)
    print(f"画布尺寸: {W} x {H}")

    nch = 3 if frames[0].ndim == 3 else 1
    acc = np.zeros((H, W, nch), np.float64)
    wacc = np.zeros((H, W), np.float64)
    wbase = blend_weight(h, w, power)

    for f, (px, py) in zip(frames, positions):
        ix, iy = int(np.floor(px)), int(np.floor(py))
        fx, fy = px - ix, py - iy
        # 亚像素平移(把小数部分烘进图像); CUBIC 比 LINEAR 少损锐度
        if fx > 1e-3 or fy > 1e-3:
            M = np.float32([[1, 0, fx], [0, 1, fy]])
            fsh = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_CUBIC)
            wsh = cv2.warpAffine(wbase, M, (w, h), flags=cv2.INTER_LINEAR)
        else:
            fsh, wsh = f, wbase
        ox, oy = ix - int(x_min), iy - int(y_min)
        fsh = fsh.reshape(h, w, nch).astype(np.float64)
        acc[oy : oy + h, ox : ox + w] += fsh * wsh[..., None]
        wacc[oy : oy + h, ox : ox + w] += wsh

    out = acc / np.maximum(wacc, 1e-9)[..., None]
    out[wacc < 1e-9] = 0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[..., 0] if nch == 1 else out


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="散斑扫描拼接")
    ap.add_argument("folder", help="帧所在文件夹(frame_*.png, 按文件名排序)")
    ap.add_argument("-o", "--output", default=None, help="输出全景图路径")
    ap.add_argument("--min-shift", type=float, default=2.0,
                    help="小于该位移(px)的帧视为滑台静止, 跳过 (默认 2)")
    ap.add_argument("--min-ncc", type=float, default=0.5,
                    help="重叠区 NCC 低于该值判为配准失败 (默认 0.5)")
    ap.add_argument("--blend-power", type=int, default=4,
                    help="融合权重次幂, 越大越接近'就近取帧', 锐度越高 (默认 4)")
    ap.add_argument("--flat", action="store_true",
                    help="平场校正: 用全帧平均估计照明分布并除掉(帧数>=10 时推荐)")
    ap.add_argument("--gray", action="store_true", help="输出灰度全景(DIC 用)")
    args = ap.parse_args()

    exts = ("*.png", "*.jpg", "*.bmp", "*.tif", "*.tiff")
    files = sorted(f for e in exts for f in glob.glob(os.path.join(args.folder, e)))
    if len(files) < 2:
        sys.exit(f"文件夹里帧数不足: {args.folder} ({len(files)} 张)")
    print(f"共 {len(files)} 帧")

    frames, grays = [], []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_COLOR)
        if img is None:
            print(f"  跳过无法读取的 {f}")
            continue
        frames.append(img)
        grays.append(img if args.gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    grays = [g.astype(np.float32) for g in grays]

    if args.flat and len(grays) >= 10:
        print("平场校正...")
        ff = flat_field(grays)
        grays = [g / ff for g in grays]
        ff3 = ff if frames[0].ndim == 2 else ff[..., None]
        frames = [np.clip(f / ff3, 0, 255).astype(np.uint8) for f in frames]

    # 逐帧配准(相对上一保留帧); 配准一律在高通图上做, 消除低频偏置
    kept_frames = [frames[0]]
    positions = [(0.0, 0.0)]
    prev_hp = highpass(grays[0])
    px = py = 0.0
    n_skip = n_fail = 0
    h, w = prev_hp.shape
    for i in range(1, len(grays)):
        cur_hp = highpass(grays[i])
        dx, dy, ncc = register_pair(prev_hp, cur_hp)
        if abs(dx) < args.min_shift and abs(dy) < args.min_shift:
            n_skip += 1
            continue  # 滑台没动
        if ncc < args.min_ncc:
            print(f"  [警告] 帧 {i} 配准质量低 (NCC={ncc:.2f}), 已跳过")
            n_fail += 1
            continue
        if abs(dx) > 0.85 * w or abs(dy) > 0.85 * h:
            print(f"  [警告] 帧 {i} 位移过大 ({dx:.0f},{dy:.0f}), 重叠不足, 已跳过")
            n_fail += 1
            continue
        px += dx
        py += dy
        positions.append((px, py))
        kept_frames.append(frames[i])
        prev_hp = cur_hp
        if len(positions) % 20 == 0:
            print(f"  已配准 {len(positions)} 帧, 当前位置 ({px:.1f}, {py:.1f}), NCC={ncc:.3f}")

    print(f"保留 {len(kept_frames)} 帧, 静止跳过 {n_skip}, 失败 {n_fail}")
    if len(kept_frames) < 2:
        sys.exit("有效帧不足, 无法拼接")

    # 相邻保留帧重叠率提示
    steps = np.diff([p[0] for p in positions]), np.diff([p[1] for p in positions])
    step = np.hypot(steps[0], steps[1])
    ovl = 1 - np.median(step) / max(w, h)
    print(f"相邻帧位移中位数 {np.median(step):.1f}px, 估计重叠率 ~{ovl*100:.0f}%")
    if ovl < 0.3:
        print("  [提示] 重叠率偏低, 建议放慢滑台或提高采集帧率")

    pano = composite(kept_frames, positions, power=args.blend_power)

    out = args.output or os.path.join(args.folder, "panorama.png")
    cv2.imwrite(out, pano)
    print(f"全景已保存: {out}  ({pano.shape[1]} x {pano.shape[0]})")

    csv_path = os.path.join(os.path.dirname(out) or ".", "positions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        wcsv = csv.writer(fp)
        wcsv.writerow(["frame_index", "x_px", "y_px"])
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        x_min, y_min = np.floor(min(xs)), np.floor(min(ys))
        for i, (x, y) in enumerate(positions):
            wcsv.writerow([i, f"{x - x_min:.3f}", f"{y - y_min:.3f}"])
    print(f"帧位置已保存: {csv_path}")


if __name__ == "__main__":
    main()
