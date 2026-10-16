# -*- coding: utf-8 -*-
"""
实时三维 DIC 核心引擎(无界面) —— 无校正(unrectified)路线。

旧长焦大汇聚角构型曾出现 stereoRectify 映射全黑，目前采用不做平面校正的路线:
  - 左右图各自去畸变(不旋转);
  - 用测量面单应 H 把右图预扭到左视角(SIFT+RANSAC 自举, 消掉汇聚角
    带来的大位移与透视差), DIS 稠密光流只匹配小残差;
  - 复合出左右稠密对应, 逐像素线性三角化得三维;
  - 位移 = 当前三维 - 参考三维, 变换到试件平面坐标系(U,V 面内, W 离面)。

精度定位: 光流级(~0.1-0.3px), 供实时云图; 定量精细分析对存盘原始帧
离线做子区 DIC。
"""
import cv2
import numpy as np


def contrast_normalize(img, win=31, floor=2.0, gain=40.0):
    """局部对比度归一化(抗反光/照明不均): (I - 局部均值)/局部标准差。

    反光 = 局部亮度加性抬升 + 增益变化, 归一化两者一并消除, 且保留
    灰度亚像素梯度(优于二值化)。等效于 ZNCC 匹配准则的光照免疫性。
    """
    f = img.astype(np.float32)
    mu = cv2.blur(f, (win, win))
    var = cv2.blur(f * f, (win, win)) - mu * mu
    sd = np.sqrt(np.maximum(var, 0))
    out = 128.0 + gain * (f - mu) / (sd + floor)
    # 离线平均图保持浮点亚灰阶；实时 Mono8 路线仍为 DIS 所需的 uint8。
    dtype = np.float32 if np.issubdtype(img.dtype, np.floating) else np.uint8
    return np.clip(out, 0, 255).astype(dtype)


def saturation_mask(img, thresh=250, min_area=400, grow=15):
    """大块反光区掩码(True=不可用)。

    单个饱和散斑白点是正常现象(不剔除); 只有面积超过 min_area 的
    饱和连通域才是镜面反光斑, 剔除并膨胀 grow px 覆盖光晕边缘。
    """
    sat = (img >= thresh).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    big_ids = [i for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not big_ids:
        return np.zeros(img.shape, bool)
    big = np.isin(lbl, big_ids).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(big, k).astype(bool)


class DicEngine:
    def __init__(self, calib_npz, out_size=(1024, 768), texture_thresh=6.0,
                 epi_thresh=1.5):
        c = np.load(calib_npz)
        W, H = int(c["image_size"][0]), int(c["image_size"][1])
        self.out_size = out_size
        s = out_size[0] / W
        S = np.diag([s, s, 1.0])
        self.K1 = S @ c["K1"]
        self.K2 = S @ c["K2"]
        self.R = c["R"]
        self.T = c["T"].reshape(3)
        # 去畸变映射(仅去畸变, 不旋转)
        self.mapL = cv2.initUndistortRectifyMap(
            c["K1"], c["D1"], None, self.K1, out_size, cv2.CV_16SC2)
        self.mapR = cv2.initUndistortRectifyMap(
            c["K2"], c["D2"], None, self.K2, out_size, cv2.CV_16SC2)
        # 本征矩阵(去畸变归一化坐标的极线约束)
        tx, ty, tz = self.T
        Tx = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        self.E = Tx @ self.R
        self.F = np.linalg.inv(self.K2).T @ self.E @ np.linalg.inv(self.K1)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.texture_thresh = texture_thresh
        self.epi_thresh = epi_thresh
        self.anti_glare = True   # 匹配前局部对比度归一化 + 饱和区剔除
        self.edge_erode = 12     # 有效区向内腐蚀(px): 剔除试件边缘的
                                 # 光流跨界污染带(边缘持续抖动的来源)
        self.H_rl = None      # 右->左 预扭单应
        self.ref = None
        w, h = out_size
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        self.gx, self.gy = gx, gy
        self._ref_acc = None

    # ---------- 基础 ----------

    def undistort_pair(self, rawL, rawR):
        l = cv2.remap(rawL, *self.mapL, cv2.INTER_LINEAR)
        r = cv2.remap(rawR, *self.mapR, cv2.INTER_LINEAR)
        return l, r

    def _bootstrap_H(self, l, r):
        """SIFT+RANSAC 求右->左单应(测量面近平面假设, 仅作预扭, 不参与三角化)。"""
        sift = cv2.SIFT_create(nfeatures=4000)
        # 仅 SIFT 特征检测使用 8 位副本，后续子区匹配保留浮点原图。
        l8 = np.clip(np.rint(l), 0, 255).astype(np.uint8)
        r8 = np.clip(np.rint(r), 0, 255).astype(np.uint8)
        k1, d1 = sift.detectAndCompute(l8, None)
        k2, d2 = sift.detectAndCompute(r8, None)
        if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
            raise RuntimeError("SIFT 特征不足, 无法自举立体对应(纹理太弱?)")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(d2, d1, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            raise RuntimeError(f"左右匹配过少({len(good)}), 无法自举")
        src = np.float32([k2[m.queryIdx].pt for m in good])
        dst = np.float32([k1[m.trainIdx].pt for m in good])
        Hrl, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hrl is None or inl.sum() < 20:
            raise RuntimeError("单应自举失败")
        m = inl.ravel().astype(bool)
        return Hrl, int(inl.sum()), len(good), dst[m], src[m]  # 左点, 右点

    def _stereo_correspond(self, l, r):
        """返回左图每像素在右图中的对应坐标 (xr, yr)。"""
        rw = cv2.warpPerspective(r, self.H_rl, self.out_size)
        f = self._pad_flow(l, rw)
        # 扭后坐标 -> 原右图坐标 (H_rl 逆变换)
        xw = self.gx + f[..., 0]
        yw = self.gy + f[..., 1]
        Hinv = np.linalg.inv(self.H_rl).astype(np.float32)
        d = Hinv[2, 0] * xw + Hinv[2, 1] * yw + Hinv[2, 2]
        xr = (Hinv[0, 0] * xw + Hinv[0, 1] * yw + Hinv[0, 2]) / d
        yr = (Hinv[1, 0] * xw + Hinv[1, 1] * yw + Hinv[1, 2]) / d
        return xr, yr

    def _pad_flow(self, a, b):
        return self.dis.calc(a, b, None)

    def _epi_err(self, xl, yl, xr, yr):
        """点到极线距离(px, 在右图中度量)。"""
        F = self.F
        a = F[0, 0] * xl + F[0, 1] * yl + F[0, 2]
        b = F[1, 0] * xl + F[1, 1] * yl + F[1, 2]
        c = F[2, 0] * xl + F[2, 1] * yl + F[2, 2]
        return np.abs(a * xr + b * yr + c) / np.sqrt(a * a + b * b + 1e-12)

    def _triangulate(self, xl, yl, xr, yr):
        """向量化线性三角化, 返回左相机系 XYZ(mm)。"""
        K1, K2 = self.K1, self.K2
        p1 = np.stack([(xl - K1[0, 2]) / K1[0, 0],
                       (yl - K1[1, 2]) / K1[1, 1],
                       np.ones_like(xl)], axis=-1)
        p2 = np.stack([(xr - K2[0, 2]) / K2[0, 0],
                       (yr - K2[1, 2]) / K2[1, 1],
                       np.ones_like(xr)], axis=-1)
        r = p1 @ self.R.T          # R·p1 (逐像素)
        t = self.T
        # z1 使 cross(p2, z1·r + t) 最小二乘为零
        c_pr = np.cross(p2, r)
        c_pt = np.cross(p2, np.broadcast_to(t, p2.shape))
        z1 = -(c_pr * c_pt).sum(-1) / ((c_pr * c_pr).sum(-1) + 1e-12)
        return p1 * z1[..., None]  # 左相机系三维点

    # ---------- 参考 ----------

    def accumulate_reference(self, rawL, rawR):
        l, r = self.undistort_pair(rawL, rawR)
        if self._ref_acc is None:
            self._ref_acc = [l.astype(np.float32), r.astype(np.float32), 1]
        else:
            self._ref_acc[0] += l
            self._ref_acc[1] += r
            self._ref_acc[2] += 1
        return self._ref_acc[2]

    def set_reference(self, rawL, rawR):
        self.accumulate_reference(rawL, rawR)
        return self.finish_reference()

    def _prep(self, img):
        return contrast_normalize(img) if self.anti_glare else img

    def finish_reference(self):
        """建立参考: 自举单应/立体对应/三维形貌/掩码/试件坐标系。
        返回诊断 dict。"""
        l = np.clip(self._ref_acc[0] / self._ref_acc[2], 0, 255).astype(np.uint8)
        r = np.clip(self._ref_acc[1] / self._ref_acc[2], 0, 255).astype(np.uint8)
        self._ref_acc = None
        lp, rp = self._prep(l), self._prep(r)
        self.H_rl, n_inl, n_match, ptsL, ptsR = self._bootstrap_H(lp, rp)
        # 标定有效性自检: SIFT 内点对标定 F 的极线残差。位姿漂移
        # (长焦下 0.006° 即 ~1.6px)会让该值远超标定 RMS。
        calib_epi = float(np.median(self._epi_err(
            ptsL[:, 0], ptsL[:, 1], ptsR[:, 0], ptsR[:, 1])))
        xr, yr = self._stereo_correspond(lp, rp)
        epi = self._epi_err(self.gx, self.gy, xr, yr)
        f32 = l.astype(np.float32)
        mu = cv2.blur(f32, (15, 15))
        std = np.sqrt(np.maximum(cv2.blur(f32 * f32, (15, 15)) - mu * mu, 0))
        w_, h_ = self.out_size
        inb = (xr >= 1) & (xr < w_ - 1) & (yr >= 1) & (yr < h_ - 1)
        mask = (std > self.texture_thresh) & (epi < self.epi_thresh) & inb
        if self.anti_glare:
            mask &= ~saturation_mask(l)
        if self.edge_erode > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.edge_erode + 1,) * 2)
            mask = cv2.erode(mask.astype(np.uint8), k).astype(bool)
        XYZ = self._triangulate(self.gx, self.gy, xr, yr)
        # 试件平面坐标系: 拟合平面 -> (e1面内x, e2面内y, n法向)
        pts = XYZ[mask]
        if len(pts) < 500:
            raise RuntimeError("有效像素过少, 参考建立失败")
        ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts[::max(1, len(pts)//5000)] - ctr,
                                 full_matrices=False)
        n = vt[2] / np.linalg.norm(vt[2])
        if n[2] > 0:
            n = -n                      # 法向朝向相机
        e1 = np.cross([0, 1, 0], n); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.basis = np.stack([e1, e2, n])   # 3x3, 行向量
        self.ref = {"l": l, "lp": lp, "XYZ": XYZ, "mask": mask}
        zc = XYZ[..., 2][mask]
        return {"epi_med": float(np.median(epi[mask])),
                "calib_epi": calib_epi,
                "valid_frac": float(mask.mean()),
                "z_range": (float(np.percentile(zc, 2)),
                            float(np.percentile(zc, 98))),
                "sift_matches": n_match, "sift_inliers": n_inl}

    # ---------- 逐帧 ----------

    def compute(self, rawL, rawR):
        if self.ref is None:
            return None
        l, r = self.undistort_pair(rawL, rawR)
        lp, rp = self._prep(l), self._prep(r)
        # 时序流: 参考左图 -> 当前左图
        t = self._pad_flow(self.ref["lp"], lp)
        xl = self.gx + t[..., 0]
        yl = self.gy + t[..., 1]
        # 当前帧立体对应(在整幅上算, 再取样到被跟踪位置)
        xr_g, yr_g = self._stereo_correspond(lp, rp)
        xr = cv2.remap(xr_g.astype(np.float32), xl, yl, cv2.INTER_LINEAR)
        yr = cv2.remap(yr_g.astype(np.float32), xl, yl, cv2.INTER_LINEAR)
        epi = self._epi_err(xl, yl, xr, yr)
        XYZc = self._triangulate(xl, yl, xr, yr)
        D = XYZc - self.ref["XYZ"]           # 左相机系位移
        # 变换到试件坐标系
        u = D @ self.basis[0]
        v = D @ self.basis[1]
        w = D @ self.basis[2]
        w_, h_ = self.out_size
        inb = (xl >= 1) & (xl < w_ - 1) & (yl >= 1) & (yl < h_ - 1)
        mask = self.ref["mask"] & inb & (epi < self.epi_thresh)
        if self.anti_glare:
            # 当前帧饱和区(反光会移动), 在被跟踪位置上取样后剔除
            satc = saturation_mask(l).astype(np.float32)
            mask &= cv2.remap(satc, xl, yl, cv2.INTER_LINEAR) < 0.3
        mag = np.sqrt(u * u + v * v + w * w)
        return {"u": u, "v": v, "w": w, "mag": mag, "mask": mask, "gray": l}
