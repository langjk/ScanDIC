# -*- coding: utf-8 -*-
"""
实时三维 DIC 云图: 双目同步采集 -> 光流级三维位移场 -> 实时彩色云图。

用法: python live_dic_gui.py   (运行前关闭 MV Viewer)
流程: 打开相机 -> 试件静止时点"设为参考" -> 加载/变形, 云图实时刷新。
U/V=面内位移, W=离面位移(试件平面坐标系), 单位 mm。
"""
import os
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from scandic.gui.calib_gui import CamThread, TriggerPump, LEFT_SN, RIGHT_SN
from scandic.dic_engine import DicEngine
from scandic.paths import PROJECT_ROOT as ROOT


def newest_calib():
    """自动取最新一次标定(重标后无需改代码)。"""
    import glob as _g
    cands = sorted(_g.glob(os.path.join(
        ROOT, "captures", "stereo_calib_*", "stereo_calib.npz")))
    if not cands:
        raise RuntimeError("captures 下没有任何 stereo_calib.npz, 请先标定")
    return cands[-1]


DEFAULT_CALIB = newest_calib()
REF_AVG = 3


class SeqRecorder(QtCore.QThread):
    """连续静态采集: 按间隔保存多帧平均的原始图像对(16bit PNG, 保留
    平均后的亚灰阶精度), 供 offline_dic.py 离线高精度子区 DIC。"""
    tick = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(str)

    def __init__(self, thL, thR, out_dir, gap_s, n_avg, calib, parent=None):
        super().__init__(parent)
        self.thL, self.thR = thL, thR
        self.out_dir = out_dir
        self.gap = gap_s
        self.n_avg = n_avg
        self.calib = calib
        self._running = False

    def _collect(self, th):
        """从相机线程收集 n_avg 个不同帧求和。"""
        acc, cnt, prev = None, 0, None
        t0 = time.time()
        while cnt < self.n_avg and self._running and time.time() - t0 < 15:
            f = th.latest()
            if f is None or f is prev:
                self.msleep(15)
                continue
            prev = f
            acc = f.astype(np.float32) if acc is None else acc + f
            cnt += 1
        return acc, cnt

    def run(self):
        import json
        self._running = True
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "meta.json"), "w",
                  encoding="utf-8") as fp:
            json.dump({"calib": self.calib, "n_avg": self.n_avg,
                       "gap_s": self.gap,
                       "start": time.strftime("%Y-%m-%d %H:%M:%S")}, fp,
                      ensure_ascii=False, indent=2)
        csv_fp = open(os.path.join(self.out_dir, "timestamps.csv"), "w",
                      encoding="utf-8")
        csv_fp.write("state,t_sec\n")
        t_start = time.time()
        idx = 0
        while self._running:
            t_next = time.time() + self.gap
            a, na = self._collect(self.thL)
            b, nb = self._collect(self.thR)
            if a is None or b is None or na < 1 or nb < 1:
                self.status.emit("序列采集: 取帧失败, 跳过本状态")
            else:
                for img, cnt, tag in ((a, na, "L"), (b, nb, "R")):
                    u16 = np.clip(img / cnt * 256.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(
                        self.out_dir, f"state_{idx:04d}_{tag}.png"), u16,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
                csv_fp.write(f"{idx},{time.time()-t_start:.3f}\n")
                csv_fp.flush()
                idx += 1
                self.tick.emit(idx)
            while self._running and time.time() < t_next:
                self.msleep(50)
        csv_fp.close()

    def stop(self):
        self._running = False
        self.wait(20000)


class DicWorker(QtCore.QThread):
    result = QtCore.pyqtSignal(dict)
    status = QtCore.pyqtSignal(str)

    def __init__(self, thL, thR, calib, parent=None):
        super().__init__(parent)
        self.thL, self.thR = thL, thR
        self.eng = DicEngine(calib)
        self._mode = "idle"     # idle / ref / run
        self._running = False
        self._ref_left = 0

    def request_reference(self):
        self._ref_left = REF_AVG
        self.eng._ref_acc = None
        self._mode = "ref"
        self.status.emit("采集参考中, 保持试件静止...")

    def run(self):
        self._running = True
        n, t0 = 0, time.time()
        while self._running:
            a = self.thL.latest()
            b = self.thR.latest()
            if a is None or b is None:
                self.msleep(50)
                continue
            if self._mode == "ref":
                self.eng.accumulate_reference(a, b)
                self._ref_left -= 1
                self.msleep(150)  # 等下一对新帧
                if self._ref_left <= 0:
                    try:
                        diag = self.eng.finish_reference()
                        msg = (f"参考已建立: 极线误差 {diag['epi_med']:.2f}px, "
                               f"标定F残差 {diag['calib_epi']:.2f}px, "
                               f"有效区 {diag['valid_frac']*100:.0f}%, "
                               f"Z {diag['z_range'][0]:.0f}~{diag['z_range'][1]:.0f}mm")
                        if diag["calib_epi"] > 0.8:
                            msg += " —— [警告] 标定F残差偏大, 相机位姿可能已变, 建议重新标定!"
                        self.status.emit(msg)
                        self._mode = "run"
                    except RuntimeError as e:
                        self.status.emit(f"参考建立失败: {e}")
                        self._mode = "idle"
                continue
            if self._mode != "run":
                self.msleep(100)
                continue
            res = self.eng.compute(a, b)
            if res is None:
                continue
            n += 1
            now = time.time()
            res["fps"] = n / (now - t0) if now > t0 else 0
            if now - t0 > 5:
                n, t0 = 0, now
            self.result.emit(res)

    def stop(self):
        self._running = False
        self.wait(8000)


class LiveDicWindow(QtWidgets.QMainWindow):
    COMPS = [("U (面内x)", "u"), ("V (面内y)", "v"),
             ("W (离面)", "w"), ("合位移", "mag")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanDIC 实时三维 DIC")
        self.thL = self.thR = None
        self.pump = None
        self.worker = None
        self.seq = None
        self.last_res = None
        self.out_dir = None
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        self.view = QtWidgets.QLabel("未开始")
        self.view.setMinimumSize(1080, 760)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet("background:#181818; color:#888;")
        v.addWidget(self.view, stretch=1)

        self.stats = QtWidgets.QLabel("")
        self.stats.setStyleSheet("font-size:15px; font-weight:bold;")
        self.stats.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.stats)

        r = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("打开相机")
        self.open_btn.clicked.connect(self.toggle_cams)
        r.addWidget(self.open_btn)
        self.ref_btn = QtWidgets.QPushButton("设为参考(零位移)")
        self.ref_btn.setEnabled(False)
        self.ref_btn.clicked.connect(lambda: self.worker.request_reference())
        r.addWidget(self.ref_btn)
        r.addWidget(QtWidgets.QLabel("分量:"))
        self.comp_combo = QtWidgets.QComboBox()
        self.comp_combo.addItems([c[0] for c in self.COMPS])
        self.comp_combo.setCurrentIndex(2)
        r.addWidget(self.comp_combo)
        self.auto_range = QtWidgets.QCheckBox("自动量程")
        self.auto_range.setChecked(True)
        r.addWidget(self.auto_range)
        self.auto_exp = QtWidgets.QCheckBox("自动曝光")
        self.auto_exp.stateChanged.connect(self.apply_auto_exp)
        r.addWidget(self.auto_exp)
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(20, 1000000)
        self.exp_spin.setValue(10000)
        self.exp_spin.setSuffix(" us")
        r.addWidget(self.exp_spin)
        b = QtWidgets.QPushButton("应用曝光")
        b.clicked.connect(self.apply_exp)
        r.addWidget(b)
        r.addWidget(QtWidgets.QLabel("固定量程±(um):"))
        self.range_spin = QtWidgets.QDoubleSpinBox()
        self.range_spin.setRange(0.1, 100000)
        self.range_spin.setValue(500.0)
        self.range_spin.setDecimals(1)
        r.addWidget(self.range_spin)
        self.glare_chk = QtWidgets.QCheckBox("抗反光")
        self.glare_chk.setChecked(True)
        self.glare_chk.stateChanged.connect(self.toggle_glare)
        r.addWidget(self.glare_chk)
        self.fill_chk = QtWidgets.QCheckBox("填补小孔")
        self.fill_chk.setChecked(True)
        self.fill_chk.setToolTip("大喷斑/反光内部无纹理测不到, 用周边数据插值"
                                 "填平仅供显示; 统计值仍只用真实测量区")
        r.addWidget(self.fill_chk)
        self.save_btn = QtWidgets.QPushButton("保存当前场")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_current)
        r.addWidget(self.save_btn)
        r.addStretch(1)
        v.addLayout(r)

        # 连续静态采集(离线高精度 DIC 用)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("连续静态采集: 间隔"))
        self.seq_gap = QtWidgets.QDoubleSpinBox()
        self.seq_gap.setRange(0.5, 3600)
        self.seq_gap.setValue(5.0)
        self.seq_gap.setSuffix(" s")
        r2.addWidget(self.seq_gap)
        r2.addWidget(QtWidgets.QLabel("每状态平均"))
        self.seq_avg = QtWidgets.QSpinBox()
        self.seq_avg.setRange(1, 16)
        self.seq_avg.setValue(4)
        self.seq_avg.setSuffix(" 帧")
        r2.addWidget(self.seq_avg)
        self.seq_btn = QtWidgets.QPushButton("开始序列采集")
        self.seq_btn.setEnabled(False)
        self.seq_btn.clicked.connect(self.toggle_seq)
        r2.addWidget(self.seq_btn)
        self.seq_label = QtWidgets.QLabel("")
        r2.addWidget(self.seq_label)
        r2.addStretch(1)
        v.addLayout(r2)

        self.log = QtWidgets.QLabel("就绪。流程: 打开相机 -> 设为参考 -> 加载变形")
        v.addWidget(self.log)

    def logmsg(self, s):
        self.log.setText(s)

    def on_cam_metrics(self, serial, m):
        # 黑底白斑试件整幅均值本来就低, 只有接近全黑才是照明/曝光故障;
        # 连续 3 次过低才报, 避免瞬时坏帧误报
        if m["mean"] < 12:
            self._dim_streak = getattr(self, "_dim_streak", 0) + 1
        else:
            self._dim_streak = 0
        if (self._dim_streak >= 3
                and time.time() - getattr(self, "_dim_warn_t", 0) > 8):
            self._dim_warn_t = time.time()
            self.logmsg(f"[警告] 画面接近全黑(亮度{m['mean']:.0f}): 检查照明/曝光")

    def _each_cam(self, fn):
        for t in (self.thL, self.thR):
            if t and t.cam:
                try:
                    fn(t.cam)
                except RuntimeError as e:
                    self.logmsg(f"[{t.serial}] {e}")

    def apply_auto_exp(self):
        on = self.auto_exp.isChecked()
        self._each_cam(lambda c: c.set_exposure_auto(on))
        self.exp_spin.setEnabled(not on)
        self.logmsg(f"自动曝光: {'开' if on else '关'} —— 曝光变化后请重新设为参考")

    def apply_exp(self):
        if self.auto_exp.isChecked():
            self.auto_exp.setChecked(False)
        v = float(self.exp_spin.value())
        self._each_cam(lambda c: c.set_exposure(v))
        self.logmsg(f"曝光 = {v:.0f} us —— 请重新设为参考")

    def toggle_glare(self):
        if self.worker is not None:
            self.worker.eng.anti_glare = self.glare_chk.isChecked()
            self.worker.eng.ref = None
            self.worker._mode = "idle"
            self.logmsg(f"抗反光: {'开' if self.glare_chk.isChecked() else '关'}"
                        " —— 预处理已变更, 请重新点\"设为参考\"")

    def toggle_cams(self):
        if self.thL is None:
            calib = DEFAULT_CALIB
            if not os.path.exists(calib):
                self.logmsg(f"找不到标定文件 {calib}")
                return
            self.thL = CamThread(LEFT_SN)
            self.thR = CamThread(RIGHT_SN)
            for t in (self.thL, self.thR):
                t.error.connect(lambda sn, m: self.logmsg(f"[{sn}] {m}"))
                t.metrics.connect(self.on_cam_metrics)
                t.start()
            self.pump = TriggerPump(self.thL, self.thR)
            self.pump.start()
            self.worker = DicWorker(self.thL, self.thR, calib)
            self.worker.result.connect(self.on_result)
            self.worker.status.connect(self.logmsg)
            self.worker.start()
            self.open_btn.setText("关闭相机")
            self.ref_btn.setEnabled(True)
            self.seq_btn.setEnabled(True)
            self.logmsg("相机已启动。试件就位且静止后点\"设为参考\"。")
        else:
            if self.seq is not None:
                self.toggle_seq()
            self.worker.stop(); self.worker = None
            self.pump.stop(); self.pump = None
            self.thL.stop(); self.thR.stop()
            self.thL = self.thR = None
            self.open_btn.setText("打开相机")
            self.ref_btn.setEnabled(False)
            self.seq_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

    def toggle_seq(self):
        if self.seq is None:
            out = os.path.join(ROOT, "captures", time.strftime("seq_%Y%m%d_%H%M%S"))
            self.seq = SeqRecorder(self.thL, self.thR, out,
                                   self.seq_gap.value(), self.seq_avg.value(),
                                   DEFAULT_CALIB)
            self.seq.tick.connect(lambda n: self.seq_label.setText(f"已存 {n} 状态"))
            self.seq.status.connect(self.logmsg)
            self.seq.start()
            self.seq_btn.setText("停止序列采集")
            self.logmsg(f"序列采集中 -> {out}  (每状态约 25MB, 注意磁盘; "
                        f"离线处理: python offline_dic.py \"{out}\")")
        else:
            self.seq.stop()
            out = self.seq.out_dir
            self.seq = None
            self.seq_btn.setText("开始序列采集")
            self.logmsg(f"序列采集已停止 -> {out}")

    # ---------- 渲染 ----------
    def on_result(self, res):
        self.last_res = res
        self.save_btn.setEnabled(True)
        key = self.COMPS[self.comp_combo.currentIndex()][1]
        f = res[key]
        m = res["mask"]
        if not m.any():
            return
        m_disp = m
        if self.fill_chk.isChecked():
            # 填补被有效区包围的小孔(大喷斑/反光内部), 仅显示层插值
            inv = (~m).astype(np.uint8)
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
            H_, W_ = m.shape
            fill_ids = [i for i in range(1, n)
                        if stats[i, cv2.CC_STAT_AREA] < 4000
                        and stats[i, cv2.CC_STAT_LEFT] > 0
                        and stats[i, cv2.CC_STAT_TOP] > 0
                        and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < W_
                        and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < H_]
            if fill_ids:
                hole = np.isin(lbl, fill_ids)
                wsum = cv2.blur((f * m).astype(np.float32), (41, 41))
                msum = cv2.blur(m.astype(np.float32), (41, 41))
                est = wsum / np.maximum(msum, 1e-6)
                f = np.where(hole, est, f)
                m_disp = m | hole
        if self.auto_range.isChecked():
            if key == "mag":
                lo, hi = 0.0, float(np.percentile(f[m], 98))
            else:
                a = float(np.percentile(np.abs(f[m]), 98))
                lo, hi = -a, a
            hi = max(hi, lo + 1e-6)
        else:
            r_ = self.range_spin.value() / 1000.0   # um -> mm
            lo, hi = (0.0, r_) if key == "mag" else (-r_, r_)
        norm = np.clip((f - lo) / (hi - lo), 0, 1)
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        g3 = cv2.cvtColor(res["gray"], cv2.COLOR_GRAY2BGR)
        vis = np.where(m_disp[..., None],
                       (0.65 * cmap + 0.35 * g3).astype(np.uint8), g3)
        # 色条 + 刻度标注(um)
        h_ = vis.shape[0]
        bar = np.linspace(1, 0, h_, dtype=np.float32)[:, None]
        bar = cv2.applyColorMap((bar * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        bar = np.repeat(bar, 48, axis=1)
        panel = np.full((h_, 150, 3), 24, np.uint8)
        panel[:, :48] = bar
        lo_u, hi_u = lo * 1000, hi * 1000
        fmt = "{:+.1f}" if (hi_u - lo_u) < 20 else "{:+.0f}"
        ticks = [(hi_u, 26), ((hi_u + lo_u) / 2, h_ // 2 + 6), (lo_u, h_ - 12)]
        cv2.putText(panel, "um", (54, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for val, ty in ticks:
            cv2.line(panel, (40, min(max(ty - 6, 0), h_ - 1)),
                     (52, min(max(ty - 6, 0), h_ - 1)), (255, 255, 255), 1)
            cv2.putText(panel, fmt.format(val), (54, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)
        vis = np.hstack([vis, np.full((h_, 8, 3), 24, np.uint8), panel])
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.view.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pix)
        vals = f[m] * 1000.0   # -> um
        self.stats.setText(
            f"{self.COMPS[self.comp_combo.currentIndex()][0]}  "
            f"量程 [{lo*1000:+.1f}, {hi*1000:+.1f}] um   "
            f"均值 {vals.mean():+.1f}  峰值 {vals[np.abs(vals).argmax()]:+.1f} um   "
            f"{res['fps']:.1f} fps")

    def save_current(self):
        if self.last_res is None:
            return
        if self.out_dir is None:
            self.out_dir = os.path.join(ROOT, "captures",
                                        time.strftime("dic_%Y%m%d_%H%M%S"))
            os.makedirs(self.out_dir, exist_ok=True)
        ts = time.strftime("%H%M%S")
        res = self.last_res
        np.savez_compressed(
            os.path.join(self.out_dir, f"field_{ts}.npz"),
            u=res["u"], v=res["v"], w=res["w"], mask=res["mask"])
        pm = self.view.pixmap()
        if pm is not None:
            pm.save(os.path.join(self.out_dir, f"cloud_{ts}.png"))
        self.logmsg(f"已保存 field_{ts}.npz / cloud_{ts}.png -> {self.out_dir}")

    def closeEvent(self, ev):
        if self.thL is not None:
            self.toggle_cams()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = LiveDicWindow()
    win.resize(1300, 980)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
