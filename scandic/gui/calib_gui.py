# -*- coding: utf-8 -*-
"""
双目标定 GUI: 实时预览 + 清晰度/亮度指标(调光圈对焦) + 自动曝光 +
圆点/棋盘检测状态 + 一键采集标定对 + 界面内跑立体标定。

用法: python calib_gui.py   (运行前关闭 MV Viewer)
"""
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from scandic.hardware.dahua_cam import DahuaCamera

LEFT_SN = "BK06664AAK00001"
RIGHT_SN = "BK24404AAK00002"
PREVIEW_FPS_CAM = 10       # 相机端限帧, 降低带宽/CPU
from scandic.paths import PROJECT_ROOT as ROOT


def make_blob_detector(scale):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 150 * scale * scale
    p.maxArea = 80000 * scale * scale
    p.filterByCircularity = True
    p.minCircularity = 0.6
    p.filterByConvexity = False
    p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)


class CamThread(QtCore.QThread):
    """持续取流, 缓存最新帧; 计算指标并发信号。"""
    metrics = QtCore.pyqtSignal(str, dict)   # serial, {sharp, mean, sat, fps}
    error = QtCore.pyqtSignal(str, str)

    def __init__(self, serial, parent=None):
        super().__init__(parent)
        self.serial = serial
        self.cam = None
        self.ready = False
        self._latest = None
        self._lock = QtCore.QMutex()
        self._running = False

    def latest(self):
        with QtCore.QMutexLocker(self._lock):
            return self._latest

    def run(self):
        try:
            self.cam = DahuaCamera.open(serial=self.serial)
            # 软触发模式: 由 TriggerPump 对两台同时发触发(时差~0.3ms),
            # 保证左右帧同一瞬间, 手持标定板晃动不再污染立体标定
            self.cam.start(soft_trigger=True)
            self.ready = True
        except RuntimeError as e:
            self.error.emit(self.serial, str(e))
            return
        self._running = True
        n, t0 = 0, time.time()
        last_metric = 0.0
        while self._running:
            img = self.cam.get_frame(1500)
            if img is None:
                continue
            with QtCore.QMutexLocker(self._lock):
                self._latest = img
            n += 1
            now = time.time()
            if now - last_metric > 0.4:
                last_metric = now
                h, w = img.shape
                roi = img[h//2-450:h//2+450, w//2-600:w//2+600]
                sharp = cv2.Laplacian(roi, cv2.CV_64F).var()
                fps = n / (now - t0) if now > t0 else 0
                if now - t0 > 3:
                    n, t0 = 0, now
                self.metrics.emit(self.serial, {
                    "sharp": sharp, "mean": float(img.mean()),
                    "sat": float((img > 250).mean() * 100), "fps": fps})
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait(5000)


class TriggerPump(QtCore.QThread):
    """以固定节奏对两台相机背靠背发软触发, 产生硬同步的帧对。"""

    def __init__(self, thL, thR, hz=PREVIEW_FPS_CAM, parent=None):
        super().__init__(parent)
        self.thL, self.thR = thL, thR
        self.period_ms = int(1000 / hz)
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if self.thL.ready and self.thR.ready:
                try:
                    self.thL.cam.send_trigger()
                    self.thR.cam.send_trigger()
                except (RuntimeError, AttributeError):
                    pass
            self.msleep(self.period_ms)

    def stop(self):
        self._running = False
        self.wait(3000)


class DetectThread(QtCore.QThread):
    """周期性在降采样图上检测标定板, 发左右检出状态和斑点数。"""
    result = QtCore.pyqtSignal(bool, bool, int, int)

    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self.gui = gui
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if not self.gui.detect_on.isChecked():
                self.msleep(300)
                continue
            pattern, kind = self.gui.pattern_spec()
            outs, blobs = [], []
            for th in (self.gui.thL, self.gui.thR):
                img = th.latest() if th else None
                if img is None:
                    outs.append(False)
                    blobs.append(0)
                    continue
                s = 0.25
                small = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
                if kind == "circles":
                    det = make_blob_detector(s)
                    blobs.append(len(det.detect(small)))
                    ok, _ = cv2.findCirclesGrid(
                        small, pattern, flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                        blobDetector=det)
                else:
                    blobs.append(-1)
                    ok, _ = cv2.findChessboardCornersSB(small, pattern)
                outs.append(bool(ok))
            self.result.emit(outs[0], outs[1], blobs[0], blobs[1])
            self.msleep(400)

    def stop(self):
        self._running = False
        self.wait(3000)


class CalibWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanDIC 双目标定")
        self.thL = self.thR = None
        self.detthread = None
        self.out_dir = None
        self.n_saved = 0
        self.last_auto = 0.0
        self.det_state = (False, False)
        self._build_ui()
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_previews)
        self.ui_timer.start(120)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)

        # 预览区
        row = QtWidgets.QHBoxLayout()
        self.views, self.metric_labels, self.det_labels = [], [], []
        for name in (f"左 {LEFT_SN}", f"右 {RIGHT_SN}"):
            col = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(name)
            title.setAlignment(QtCore.Qt.AlignCenter)
            col.addWidget(title)
            view = QtWidgets.QLabel("未连接")
            view.setMinimumSize(620, 460)
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setStyleSheet("background:#202020; color:#888;")
            col.addWidget(view, stretch=1)
            m = QtWidgets.QLabel("清晰度 -    亮度 -    饱和 -")
            m.setStyleSheet("font-size:15px; font-weight:bold;")
            m.setAlignment(QtCore.Qt.AlignCenter)
            col.addWidget(m)
            d = QtWidgets.QLabel("检测: -")
            d.setAlignment(QtCore.Qt.AlignCenter)
            d.setStyleSheet("background:#555; color:white; padding:3px; border-radius:3px;")
            col.addWidget(d)
            self.views.append(view)
            self.metric_labels.append(m)
            self.det_labels.append(d)
            row.addLayout(col)
        v.addLayout(row, stretch=1)

        # 相机/曝光控制
        r1 = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("打开相机")
        self.open_btn.clicked.connect(self.toggle_cams)
        r1.addWidget(self.open_btn)
        self.auto_exp = QtWidgets.QCheckBox("自动曝光")
        self.auto_exp.stateChanged.connect(self.apply_auto_exp)
        r1.addWidget(self.auto_exp)
        r1.addWidget(QtWidgets.QLabel("手动曝光(us):"))
        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(20, 1000000)
        self.exp_spin.setValue(10000)
        self.exp_spin.setSingleStep(500)
        r1.addWidget(self.exp_spin)
        b = QtWidgets.QPushButton("应用曝光")
        b.clicked.connect(self.apply_manual_exp)
        r1.addWidget(b)
        b = QtWidgets.QPushButton("读取当前曝光")
        b.clicked.connect(self.read_exposure)
        r1.addWidget(b)
        r1.addStretch(1)
        v.addLayout(r1)

        # 标定板参数 + 采集
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("板型:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["圆点阵", "棋盘格"])
        r2.addWidget(self.type_combo)
        r2.addWidget(QtWidgets.QLabel("列:"))
        self.cols_spin = QtWidgets.QSpinBox(); self.cols_spin.setRange(3, 30); self.cols_spin.setValue(11)
        r2.addWidget(self.cols_spin)
        r2.addWidget(QtWidgets.QLabel("行:"))
        self.rows_spin = QtWidgets.QSpinBox(); self.rows_spin.setRange(3, 30); self.rows_spin.setValue(9)
        r2.addWidget(self.rows_spin)
        r2.addWidget(QtWidgets.QLabel("中心距(mm):"))
        self.square_spin = QtWidgets.QDoubleSpinBox()
        self.square_spin.setRange(0.1, 200); self.square_spin.setValue(10.0); self.square_spin.setDecimals(3)
        r2.addWidget(self.square_spin)
        self.detect_on = QtWidgets.QCheckBox("实时检测")
        self.detect_on.setChecked(True)
        r2.addWidget(self.detect_on)
        r2.addStretch(1)
        v.addLayout(r2)

        r3 = QtWidgets.QHBoxLayout()
        self.cap_btn = QtWidgets.QPushButton("拍摄一对")
        self.cap_btn.setMinimumHeight(36)
        self.cap_btn.setEnabled(False)
        self.cap_btn.clicked.connect(self.capture_pair)
        r3.addWidget(self.cap_btn)
        self.auto_cap = QtWidgets.QCheckBox("自动采集(双目检出即拍, 间隔")
        r3.addWidget(self.auto_cap)
        self.auto_gap = QtWidgets.QDoubleSpinBox()
        self.auto_gap.setRange(0.5, 10); self.auto_gap.setValue(2.0)
        r3.addWidget(self.auto_gap)
        r3.addWidget(QtWidgets.QLabel("秒)"))
        self.count_label = QtWidgets.QLabel("已采 0 对")
        self.count_label.setStyleSheet("font-weight:bold;")
        r3.addWidget(self.count_label)
        self.calib_btn = QtWidgets.QPushButton("运行立体标定")
        self.calib_btn.setEnabled(False)
        self.calib_btn.clicked.connect(self.run_calibration)
        r3.addWidget(self.calib_btn)
        r3.addStretch(1)
        v.addLayout(r3)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        v.addWidget(self.log)

    def pattern_spec(self):
        kind = "circles" if self.type_combo.currentIndex() == 0 else "chess"
        return (self.cols_spin.value(), self.rows_spin.value()), kind

    def logmsg(self, s):
        self.log.appendPlainText(s)

    # ---------------- 相机 ----------------
    def toggle_cams(self):
        if self.thL is None:
            self.thL = CamThread(LEFT_SN)
            self.thR = CamThread(RIGHT_SN)
            for t in (self.thL, self.thR):
                t.metrics.connect(self.on_metrics)
                t.error.connect(lambda sn, msg: self.logmsg(f"[{sn}] 错误: {msg}"))
                t.start()
            self.pump = TriggerPump(self.thL, self.thR)
            self.pump.start()
            self.detthread = DetectThread(self)
            self.detthread.result.connect(self.on_detect)
            self.detthread.start()
            self.open_btn.setText("关闭相机")
            self.cap_btn.setEnabled(True)
            self.logmsg("相机已启动(软触发同步对拍 10Hz)。调焦看清晰度数值, 越大越清晰。")
        else:
            self.detthread.stop(); self.detthread = None
            self.pump.stop(); self.pump = None
            self.thL.stop(); self.thR.stop()
            self.thL = self.thR = None
            self.open_btn.setText("打开相机")
            self.cap_btn.setEnabled(False)
            for v_ in self.views:
                v_.setText("未连接"); v_.setPixmap(QtGui.QPixmap())

    def _each_cam(self, fn, desc):
        for t in (self.thL, self.thR):
            if t and t.cam:
                try:
                    fn(t.cam)
                except RuntimeError as e:
                    self.logmsg(f"[{t.serial}] {desc}失败: {e}")

    def apply_auto_exp(self):
        on = self.auto_exp.isChecked()
        self._each_cam(lambda c: c.set_exposure_auto(on), "自动曝光")
        self.exp_spin.setEnabled(not on)
        self.logmsg(f"自动曝光: {'开' if on else '关'}")

    def apply_manual_exp(self):
        if self.auto_exp.isChecked():
            self.auto_exp.setChecked(False)
        val = float(self.exp_spin.value())
        self._each_cam(lambda c: c.set_exposure(val), "设曝光")
        self.logmsg(f"曝光 = {val:.0f} us")

    def read_exposure(self):
        for t in (self.thL, self.thR):
            if t and t.cam:
                try:
                    self.logmsg(f"[{t.serial}] 当前曝光 {t.cam.get_double('ExposureTime'):.0f} us")
                except RuntimeError as e:
                    self.logmsg(f"[{t.serial}] 读曝光失败: {e}")

    # ---------------- 显示 ----------------
    def refresh_previews(self):
        for i, t in enumerate((self.thL, self.thR)):
            if t is None:
                continue
            img = t.latest()
            if img is None:
                continue
            small = cv2.resize(img, None, fx=0.18, fy=0.18,
                               interpolation=cv2.INTER_AREA)
            h, w = small.shape
            qimg = QtGui.QImage(small.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.views[i].size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation)
            self.views[i].setPixmap(pix)
        # 自动采集
        if (self.auto_cap.isChecked() and self.cap_btn.isEnabled()
                and all(self.det_state)
                and time.time() - self.last_auto > self.auto_gap.value()):
            self.last_auto = time.time()
            self.capture_pair()

    def on_metrics(self, serial, m):
        i = 0 if serial == LEFT_SN else 1
        self.metric_labels[i].setText(
            f"清晰度 {m['sharp']:.0f}    亮度 {m['mean']:.0f}    "
            f"饱和 {m['sat']:.2f}%    {m['fps']:.1f}fps")
        # 饱和过高标红
        style = "font-size:15px; font-weight:bold;"
        if m["sat"] > 1.0:
            style += "color:#c62828;"
        self.metric_labels[i].setStyleSheet(style)

    def on_detect(self, okL, okR, nL, nR):
        self.det_state = (okL, okR)
        (cols, rows), kind = self.pattern_spec()
        need = cols * rows
        for i, (ok, n) in enumerate(((okL, nL), (okR, nR))):
            if ok:
                txt = "检测: 检出 OK"
            elif kind == "circles":
                txt = f"检测: 未检出 (斑点{n}个, 需{cols}x{rows}={need})"
            else:
                txt = "检测: 未检出"
            self.det_labels[i].setText(txt)
            self.det_labels[i].setStyleSheet(
                f"background:{'#2e7d32' if ok else '#8b3a3a'}; color:white; "
                "padding:3px; border-radius:3px;")

    # ---------------- 采集 / 标定 ----------------
    def capture_pair(self):
        a = self.thL.latest() if self.thL else None
        b = self.thR.latest() if self.thR else None
        if a is None or b is None:
            self.logmsg("取帧失败")
            return
        if not all(self.det_state):
            self.logmsg("未保存: 左右目未同时检出标定板")
            return
        if self.out_dir is None:
            self.out_dir = os.path.join(ROOT, "captures",
                                        time.strftime("stereo_calib_%Y%m%d_%H%M%S"))
            os.makedirs(self.out_dir, exist_ok=True)
            self.logmsg(f"输出目录: {self.out_dir}")
        cv2.imwrite(os.path.join(self.out_dir, f"left_{self.n_saved:02d}.png"), a)
        cv2.imwrite(os.path.join(self.out_dir, f"right_{self.n_saved:02d}.png"), b)
        self.n_saved += 1
        self.count_label.setText(f"已采 {self.n_saved} 对")
        self.calib_btn.setEnabled(self.n_saved >= 8)
        self.logmsg(f"已保存 #{self.n_saved:02d} —— 请换一个姿态")

    def run_calibration(self):
        if not self.out_dir:
            return
        (cols, rows), kind = self.pattern_spec()
        cmd = [sys.executable, "-m", "scandic.calibration.stereo_calibrate",
               self.out_dir, "--type", kind, "--cols", str(cols),
               "--rows", str(rows), "--square", str(self.square_spin.value())]
        self.logmsg("运行: " + " ".join(cmd[3:]))
        self.calib_btn.setEnabled(False)

        def work():
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            # -m 需在项目根为 cwd 才能定位 scandic 包
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env,
                               cwd=str(ROOT))
            return (r.stdout or "") + ("\n" + r.stderr if r.returncode else "")

        class Runner(QtCore.QThread):
            done = QtCore.pyqtSignal(str)
            def run(self):
                self.done.emit(work())

        self._runner = Runner()
        self._runner.done.connect(lambda out: (
            self.logmsg(out), self.calib_btn.setEnabled(True)))
        self._runner.start()

    def closeEvent(self, ev):
        if self.thL is not None:
            self.toggle_cams()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CalibWindow()
    win.resize(1380, 940)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
