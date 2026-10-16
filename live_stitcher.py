# -*- coding: utf-8 -*-
"""
实时增量拼接核心(无界面, 供 scan_gui.py 调用, 也可离线测试)

滑台不可控, 采用"跟踪 + 关键帧"策略:
  - 每来一帧, 在降采样高通图上与上一关键帧做散斑相位相关, 快速估计位移(跟踪);
  - 位移超过阈值(默认 35% 视场)时, 全分辨率亚像素精配, 升级为关键帧;
  - 跟踪相关系数过低时报"丢失"(滑台过快/无纹理/失焦)。

实时性设计: 扫描过程中只做轻量工作 ——
  - 关键帧 PNG 由后台线程存盘(不阻塞配准);
  - 实时预览拼在 1/4 缩放的小画布上(每帧几毫秒);
  - 全分辨率羽化融合放到 finish() 一次完成, 并用整个扫描累积的
    照明均值做平场校正(散斑随机移动, 均值即照明分布)。

配准一律在高通滤波图上做, 消除暗角/照明梯度造成的亚像素偏置。
"""
import csv
import os
import queue
import threading

import cv2
import numpy as np

from stitch import register_pair, feather_weight, blend_weight, highpass


class LiveStitcher:
    def __init__(
        self,
        keep_frac=0.35,     # 位移达到视场的该比例时落一个关键帧
        min_shift=2.0,      # 小于此位移(px)视为滑台静止
        min_ncc=0.45,       # 配准质量下限(高通图上的重叠区 NCC)
        feather=120,        # 最终融合羽化宽度
        track_ds=0.5,       # 跟踪用降采样倍率
        preview_scale=0.25, # 实时预览画布缩放
        save_dir=None,      # 若给出, 关键帧实时存盘(后台线程), 全景/CSV 存这里
    ):
        self.keep_frac = keep_frac
        self.min_shift = min_shift
        self.min_ncc = min_ncc
        self.feather = feather
        self.ds = track_ds
        self.ps = preview_scale
        self.save_dir = save_dir

        self.positions = []       # 各关键帧位置(亚像素, 全景坐标)
        self.n_frames = 0
        self.n_lost = 0
        self._kf_hp = None
        self._kf_hp_ds = None
        self._kf_pos = (0.0, 0.0)
        self._tail = None         # 最后一个未达关键帧阈值的跟踪帧(收尾时补拼)
        self._frames_mem = []     # save_dir 为 None 时关键帧存内存
        self._key_paths = []
        # 照明累积(用于 finish 时平场)
        self._illum_acc = None
        self._illum_n = 0
        # 预览小画布
        self._pacc = None
        self._pwacc = None
        self._porigin = (0, 0)
        self._pbbox = None
        self._pwbase = None
        # 后台写盘
        self._wq = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._wq = queue.Queue(maxsize=8)
            self._wt = threading.Thread(target=self._write_loop, daemon=True)
            self._wt.start()

    # ---------- 对外接口 ----------

    def add_frame(self, frame_bgr):
        """喂入一帧, 返回状态 dict (state: first/static/tracking/keyframe/lost)。"""
        self.n_frames += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = highpass(gray)

        if self._kf_hp is None:
            self._set_keyframe(frame_bgr, hp, (0.0, 0.0))
            return self._status("first", 0, 0, 1.0)

        hp_ds = cv2.resize(hp, None, fx=self.ds, fy=self.ds,
                           interpolation=cv2.INTER_AREA)
        dxd, dyd, ncc = register_pair(self._kf_hp_ds, hp_ds)
        dx, dy = dxd / self.ds, dyd / self.ds

        if ncc < self.min_ncc:
            dx, dy, ncc = register_pair(self._kf_hp, hp)  # 全分辨率重试(带模板回退)
            if ncc < self.min_ncc:
                self.n_lost += 1
                return self._status("lost", dx, dy, ncc)

        if abs(dx) < self.min_shift and abs(dy) < self.min_shift:
            self._tail = None  # 回到关键帧附近, 尾巴作废
            return self._status("static", dx, dy, ncc)

        # 移动中的帧参与照明估计(finish 时做平场)
        if self._illum_acc is None:
            self._illum_acc = np.zeros_like(gray)
        self._illum_acc += gray
        self._illum_n += 1

        h, w = gray.shape
        if abs(dx) < self.keep_frac * w and abs(dy) < self.keep_frac * h:
            self._tail = (frame_bgr, hp)
            return self._status("tracking", dx, dy, ncc)

        dx, dy, ncc = register_pair(self._kf_hp, hp)  # 亚像素精配
        if ncc < self.min_ncc:
            self.n_lost += 1
            return self._status("lost", dx, dy, ncc)
        pos = (self._kf_pos[0] + dx, self._kf_pos[1] + dy)
        self._set_keyframe(frame_bgr, hp, pos)
        return self._status("keyframe", dx, dy, ncc)

    def preview(self, max_w=1400):
        """实时预览缩略图(uint8 BGR), 无内容时返回 None。"""
        if self._pbbox is None:
            return None
        x0, y0, x1, y1 = self._pbbox
        ox, oy = self._porigin
        acc = self._pacc[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        wacc = self._pwacc[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        out = acc / np.maximum(wacc, 1e-9)[..., None]
        out = np.clip(out, 0, 255).astype(np.uint8)
        if out.shape[1] > max_w:
            s = max_w / out.shape[1]
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return out

    def flat_field(self):
        """整个扫描的照明分布估计(均值 1), 样本不足时返回 None。"""
        if self._illum_n < 10:
            return None
        avg = (self._illum_acc / self._illum_n).astype(np.float32)
        avg = cv2.GaussianBlur(avg, (0, 0), 31)
        avg /= avg.mean()
        np.clip(avg, 0.3, 3.0, out=avg)
        return avg

    def finish(self):
        """全分辨率羽化融合出全景。返回 (pano uint8 BGR, 位置列表)。

        save_dir 给出时保存 panorama.png / positions.csv。
        耗时与关键帧数成正比(每帧几十毫秒量级), 在扫描结束后调用。
        """
        self._flush_tail()
        if self._wq is not None:
            self._wq.put(None)
            self._wt.join(timeout=30)
            self._wq = None
        if not self.positions:
            return None, []

        flat = self.flat_field()
        xs = np.array([p[0] for p in self.positions])
        ys = np.array([p[1] for p in self.positions])

        def load(i):
            if self._frames_mem:
                return self._frames_mem[i]
            return cv2.imread(self._key_paths[i], cv2.IMREAD_COLOR)

        f0 = load(0)
        h, w = f0.shape[:2]
        x_min, y_min = np.floor(xs.min()), np.floor(ys.min())
        W = int(np.ceil(xs.max() + w) - x_min)
        H = int(np.ceil(ys.max() + h) - y_min)
        acc = np.zeros((H, W, 3), np.float32)
        wacc = np.zeros((H, W), np.float32)
        # 中心占优融合: 保源图锐度(宽区 50/50 平均会因残余错位把散斑抹虚)
        wbase = blend_weight(h, w, power=4).astype(np.float32)

        for i, (px, py) in enumerate(self.positions):
            img = load(i).astype(np.float32)
            if flat is not None:
                img /= flat[..., None]
            ix, iy = int(np.floor(px)), int(np.floor(py))
            fx, fy = px - ix, py - iy
            if fx > 1e-3 or fy > 1e-3:
                M = np.float32([[1, 0, fx], [0, 1, fy]])
                fsh = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
                wsh = cv2.warpAffine(wbase, M, (w, h), flags=cv2.INTER_LINEAR)
            else:
                fsh, wsh = img, wbase
            ox, oy = ix - int(x_min), iy - int(y_min)
            acc[oy : oy + h, ox : ox + w] += fsh * wsh[..., None]
            wacc[oy : oy + h, ox : ox + w] += wsh

        pano = acc / np.maximum(wacc, 1e-9)[..., None]
        pano = np.clip(pano, 0, 255).astype(np.uint8)
        norm_pos = [(px - x_min, py - y_min) for px, py in self.positions]

        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, "panorama.png"), pano)
            with open(os.path.join(self.save_dir, "positions.csv"), "w",
                      newline="", encoding="utf-8") as fp:
                wcsv = csv.writer(fp)
                wcsv.writerow(["keyframe_index", "x_px", "y_px"])
                for i, (x, y) in enumerate(norm_pos):
                    wcsv.writerow([i, f"{x:.3f}", f"{y:.3f}"])
        return pano, norm_pos

    # ---------- 内部 ----------

    def _status(self, state, dx, dy, ncc):
        return {
            "state": state, "dx": dx, "dy": dy, "ncc": ncc,
            "pos": (self._kf_pos[0] + dx, self._kf_pos[1] + dy),
            "n_key": len(self.positions), "n_frames": self.n_frames,
            "n_lost": self.n_lost,
        }

    def _flush_tail(self):
        """扫描结束时, 把最后一段未达阈值的位移补成关键帧(避免尾部丢失)。"""
        if self._tail is None or self._kf_hp is None:
            return
        frame, hp = self._tail
        self._tail = None
        dx, dy, ncc = register_pair(self._kf_hp, hp)
        if ncc < self.min_ncc or (abs(dx) < 25 and abs(dy) < 25):
            return
        self._set_keyframe(frame, hp,
                           (self._kf_pos[0] + dx, self._kf_pos[1] + dy))

    def _write_loop(self):
        while True:
            item = self._wq.get()
            if item is None:
                return
            path, img = item
            # 散斑高熵, 高压缩级别极慢且省不了多少, 用最快级别
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def _set_keyframe(self, frame_bgr, hp, pos):
        self._kf_hp = hp
        self._kf_hp_ds = cv2.resize(hp, None, fx=self.ds, fy=self.ds,
                                    interpolation=cv2.INTER_AREA)
        self._kf_pos = pos
        self.positions.append(pos)
        if self._wq is not None:
            path = os.path.join(self.save_dir,
                                f"key_{len(self.positions)-1:05d}.png")
            self._key_paths.append(path)
            self._wq.put((path, frame_bgr.copy()))
        else:
            self._frames_mem.append(frame_bgr.copy())
        self._compose_preview(frame_bgr, pos)

    def _compose_preview(self, frame_bgr, pos):
        small = cv2.resize(frame_bgr, None, fx=self.ps, fy=self.ps,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        h, w = small.shape[:2]
        if self._pwbase is None:
            self._pwbase = feather_weight(
                h, w, max(8, int(self.feather * self.ps))).astype(np.float32)
        ix = int(round(pos[0] * self.ps))
        iy = int(round(pos[1] * self.ps))
        self._ensure_pcanvas(ix, iy, w, h)
        ox, oy = self._porigin
        cx, cy = ix - ox, iy - oy
        self._pacc[cy : cy + h, cx : cx + w] += small * self._pwbase[..., None]
        self._pwacc[cy : cy + h, cx : cx + w] += self._pwbase
        if self._pbbox is None:
            self._pbbox = [ix, iy, ix + w, iy + h]
        else:
            b = self._pbbox
            self._pbbox = [min(b[0], ix), min(b[1], iy),
                           max(b[2], ix + w), max(b[3], iy + h)]

    def _ensure_pcanvas(self, ix, iy, w, h, margin=1024):
        if self._pacc is None:
            self._porigin = (ix, iy)
            self._pacc = np.zeros((h, w, 3), np.float32)
            self._pwacc = np.zeros((h, w), np.float32)
            return
        ox, oy = self._porigin
        H, W = self._pwacc.shape
        nx0 = min(ox, ix - (margin if ix < ox else 0))
        ny0 = min(oy, iy - (margin if iy < oy else 0))
        nx1 = max(ox + W, ix + w + (margin if ix + w > ox + W else 0))
        ny1 = max(oy + H, iy + h + (margin if iy + h > oy + H else 0))
        if (nx0, ny0, nx1, ny1) == (ox, oy, ox + W, oy + H):
            return
        acc = np.zeros((ny1 - ny0, nx1 - nx0, 3), np.float32)
        wacc = np.zeros((ny1 - ny0, nx1 - nx0), np.float32)
        acc[oy - ny0 : oy - ny0 + H, ox - nx0 : ox - nx0 + W] = self._pacc
        wacc[oy - ny0 : oy - ny0 + H, ox - nx0 : ox - nx0 + W] = self._pwacc
        self._pacc, self._pwacc = acc, wacc
        self._porigin = (nx0, ny0)
