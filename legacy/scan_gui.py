# -*- coding: utf-8 -*-
"""
滑台实时扫描拼接程序(滑台不可控, 扫描过程中自动配准、实时拼接)

用法:
    python scan_gui.py

流程:
  1. 打开相机(序号 1 = 外接 UVC), 用清晰度指标调焦, 锁定曝光
  2. 点"开始扫描", 然后让滑台滑动 —— 界面下方实时长出拼接全景
  3. 状态灯: 绿=正常跟踪/落关键帧, 黄=静止, 红=配准丢失(滑台过快/失焦)
  4. 扫完点"结束扫描", 自动保存 panorama.png / positions.csv / 各关键帧
"""
import os
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import legacy.calib as calib
from legacy.capture_gui import CameraThread, CAPTURE_ROOT
from legacy.live_stitcher import LiveStitcher


def np2pixmap(img_bgr, target: QtWidgets.QLabel):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg).scaled(
        target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


class ScanWorker(QtCore.QThread):
    """独立线程跑配准/拼接, 始终只处理最新一帧, 处理不过来就丢帧。"""
    status = QtCore.pyqtSignal(dict)
    pano = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, stitcher, parent=None):
        super().__init__(parent)
        self.stitcher = stitcher
        self._latest = None
        self._lock = QtCore.QMutex()
        self._running = False
        self._maps = calib.load_maps()  # 有 undistort.json 则自动去畸变
        self.crop = 0.6                 # 只用画面中央区域(广角边角软/畸变大)

    def submit(self, frame):
        with QtCore.QMutexLocker(self._lock):
            self._latest = frame

    def run(self):
        self._running = True
        last_pano_t = 0.0
        while self._running:
            with QtCore.QMutexLocker(self._lock):
                f, self._latest = self._latest, None
            if f is None:
                self.msleep(5)
                continue
            if self._maps is not None and f.shape[1] == self._maps[2][0]:
                f = calib.undistort(f, self._maps)
            c = self.crop
            h, w = f.shape[:2]
            y0 = int(h * (1 - c) / 2) & ~1
            x0 = int(w * (1 - c) / 2) & ~1
            f = np.ascontiguousarray(
                f[y0 : y0 + (int(h * c) & ~1), x0 : x0 + (int(w * c) & ~1)])
            info = self.stitcher.add_frame(f)
            self.status.emit(info)
            now = time.time()
            if info["state"] in ("first", "keyframe") or now - last_pano_t > 1.0:
                p = self.stitcher.preview(max_w=1600)
                if p is not None:
                    self.pano.emit(p)
                    last_pano_t = now

    def stop(self):
        self._running = False
        self.wait(5000)


class FinishThread(QtCore.QThread):
    """结束时的全分辨率融合在后台跑, 避免界面卡死。"""
    done = QtCore.pyqtSignal(object, object)

    def __init__(self, stitcher, parent=None):
        super().__init__(parent)
        self.stitcher = stitcher

    def run(self):
        pano, pos = self.stitcher.finish()
        self.done.emit(pano, pos)


class ScanWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanDIC 实时扫描拼接")
        self.cam = None
        self.worker = None
        self.run_dir = None
        self._sharp_t = 0.0
        self._sharp = 0.0
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # 相机预览
        self.preview = QtWidgets.QLabel("未打开相机")
        self.preview.setMinimumSize(800, 450)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:#202020; color:#aaa;")
        vbox.addWidget(self.preview, stretch=3)

        # 实时全景
        vbox.addWidget(QtWidgets.QLabel("实时拼接全景:"))
        self.pano_view = QtWidgets.QLabel("(开始扫描后显示)")
        self.pano_view.setMinimumHeight(180)
        self.pano_view.setAlignment(QtCore.Qt.AlignCenter)
        self.pano_view.setStyleSheet("background:#181818; color:#888;")
        vbox.addWidget(self.pano_view, stretch=2)

        # 控制行 1: 相机
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("相机序号:"))
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.addItems(["0", "1", "2"])
        self.cam_combo.setCurrentText("1")
        row1.addWidget(self.cam_combo)
        row1.addWidget(QtWidgets.QLabel("分辨率:"))
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(["1920x1080", "1280x720", "640x480"])
        row1.addWidget(self.res_combo)
        self.open_btn = QtWidgets.QPushButton("打开相机")
        self.open_btn.clicked.connect(self.toggle_camera)
        row1.addWidget(self.open_btn)
        self.auto_exp = QtWidgets.QCheckBox("自动曝光")
        self.auto_exp.setChecked(True)
        self.auto_exp.stateChanged.connect(self.apply_exposure)
        row1.addWidget(self.auto_exp)
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(-13, 0)
        self.exp_slider.setValue(-6)
        self.exp_slider.setMaximumWidth(160)
        self.exp_slider.valueChanged.connect(self.apply_exposure)
        row1.addWidget(self.exp_slider)
        row1.addStretch(1)
        vbox.addLayout(row1)

        # 控制行 2: 扫描
        row2 = QtWidgets.QHBoxLayout()
        self.scan_btn = QtWidgets.QPushButton("开始扫描")
        self.scan_btn.setEnabled(False)
        self.scan_btn.setMinimumHeight(36)
        self.scan_btn.clicked.connect(self.toggle_scan)
        row2.addWidget(self.scan_btn)
        self.state_lamp = QtWidgets.QLabel("  待机  ")
        self.state_lamp.setStyleSheet(
            "background:#555; color:white; padding:6px; border-radius:4px;")
        row2.addWidget(self.state_lamp)
        self.info_label = QtWidgets.QLabel("")
        row2.addWidget(self.info_label, stretch=1)
        vbox.addLayout(row2)

        self.result_label = QtWidgets.QLabel("")
        self.result_label.setWordWrap(True)
        vbox.addWidget(self.result_label)

    # ---- 相机 ----
    def toggle_camera(self):
        if self.cam is None:
            idx = int(self.cam_combo.currentText())
            w, h = map(int, self.res_combo.currentText().split("x"))
            self.cam = CameraThread(idx, w, h)
            self.cam.frame_ready.connect(self.on_frame)
            self.cam.status.connect(self.statusBar().showMessage)
            self.cam.start()
            self.open_btn.setText("关闭相机")
            self.scan_btn.setEnabled(True)
            QtCore.QTimer.singleShot(500, self.apply_exposure)
        else:
            if self.worker is not None:
                self.toggle_scan()
            self.cam.stop()
            self.cam = None
            self.open_btn.setText("打开相机")
            self.scan_btn.setEnabled(False)
            self.preview.setText("未打开相机")
            self.preview.setPixmap(QtGui.QPixmap())

    def apply_exposure(self):
        self.exp_slider.setEnabled(not self.auto_exp.isChecked())
        if self.cam is not None:
            self.cam.set_exposure(self.auto_exp.isChecked(), self.exp_slider.value())

    # ---- 扫描 ----
    def toggle_scan(self):
        if self.worker is None:
            self.run_dir = os.path.join(CAPTURE_ROOT, time.strftime("scan_%Y%m%d_%H%M%S"))
            st = LiveStitcher(save_dir=self.run_dir)
            self.worker = ScanWorker(st)
            self.worker.status.connect(self.on_scan_status)
            self.worker.pano.connect(self.on_pano)
            self.worker.start()
            self.scan_btn.setText("结束扫描")
            self.result_label.setText("")
            self.info_label.setText("扫描中, 让滑台滑动...")
        else:
            self.worker.stop()
            st = self.worker.stitcher
            self.worker = None
            self.scan_btn.setText("开始扫描")
            self.scan_btn.setEnabled(False)
            self._set_lamp("融合中...", "#1565c0")
            self.info_label.setText("正在做全分辨率融合...")
            self._finisher = FinishThread(st)
            self._finisher.done.connect(self.on_finish_done)
            self._finisher.start()

    def on_finish_done(self, pano, pos):
        st = self._finisher.stitcher
        self._finisher = None
        self.scan_btn.setEnabled(self.cam is not None)
        self._set_lamp("待机", "#555")
        self.info_label.setText("")
        if pano is None:
            self.result_label.setText("没有拼到任何关键帧")
            return
        self.on_pano(pano)
        self.result_label.setText(
            f"完成: {len(pos)} 个关键帧, 全景 {pano.shape[1]}x{pano.shape[0]}px, "
            f"丢失 {st.n_lost} 次\n已保存到 {self.run_dir}\\panorama.png "
            f"(含 positions.csv 和关键帧, 可用 stitch.py 离线重拼)")

    def on_scan_status(self, info):
        s = info["state"]
        color, text = {
            "first":    ("#2e7d32", "起始帧"),
            "keyframe": ("#2e7d32", "落关键帧"),
            "tracking": ("#2e7d32", "跟踪中"),
            "static":   ("#b8860b", "滑台静止"),
            "lost":     ("#c62828", "配准丢失!"),
        }[s]
        self._set_lamp(text, color)
        self.info_label.setText(
            f"关键帧 {info['n_key']}  位置 ({info['pos'][0]:.0f}, {info['pos'][1]:.0f})  "
            f"位移 ({info['dx']:.0f}, {info['dy']:.0f})  NCC={info['ncc']:.2f}  "
            f"已处理 {info['n_frames']} 帧  丢失 {info['n_lost']}")
        if s == "lost":
            self.info_label.setStyleSheet("color:#c62828;")
        else:
            self.info_label.setStyleSheet("")

    def _set_lamp(self, text, color):
        self.state_lamp.setText(f"  {text}  ")
        self.state_lamp.setStyleSheet(
            f"background:{color}; color:white; padding:6px; border-radius:4px;")

    def on_pano(self, pano):
        self.pano_view.setPixmap(np2pixmap(pano, self.pano_view))

    # ---- 帧回调 ----
    def on_frame(self, frame):
        if self.worker is not None:
            self.worker.submit(frame)
        now = time.time()
        if now - self._sharp_t > 0.5:
            self._sharp_t = now
            h, w = frame.shape[:2]
            roi = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            self._sharp = cv2.Laplacian(g, cv2.CV_64F).var()
            self.statusBar().showMessage(
                f"{w}x{h}  清晰度={self._sharp:.0f} (调焦使其最大)")
        self.preview.setPixmap(np2pixmap(frame, self.preview))

    def closeEvent(self, ev):
        if self.worker is not None:
            self.worker.stop()
        if self.cam is not None:
            self.cam.stop()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ScanWindow()
    win.resize(1250, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
