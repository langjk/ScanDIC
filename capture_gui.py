# -*- coding: utf-8 -*-
"""
滑台扫描采集程序（免驱 UVC 相机）

用法:
    python capture_gui.py

界面功能:
  - 选择相机序号 / 分辨率, 打开实时预览
  - 锁定曝光(扫描拼接必须锁定, 否则帧间亮度不一致会产生条带)
  - 清晰度指标(Laplacian 方差, 调焦时越大越清晰)
  - 开始/停止录制: 帧按序号存为 PNG 到 captures/run_YYYYmmdd_HHMMSS/
  - 可设置保存间隔(每 N 帧存 1 帧), 控制相邻帧重叠率

录完后用 stitch.py 拼接:
    python stitch.py captures/run_xxxx -o panorama.png
"""
import os
import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

CAPTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


class CameraThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)

    def __init__(self, cam_index, width, height, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self._running = False
        self._recording = False
        self._save_dir = None
        self._save_every = 1
        self._frame_no = 0
        self._saved_no = 0
        self._cap = None
        self._lock = QtCore.QMutex()

    # ---- 控制接口(主线程调用) ----
    def start_recording(self, save_dir, save_every):
        with QtCore.QMutexLocker(self._lock):
            os.makedirs(save_dir, exist_ok=True)
            self._save_dir = save_dir
            self._save_every = max(1, int(save_every))
            self._frame_no = 0
            self._saved_no = 0
            self._recording = True

    def stop_recording(self):
        with QtCore.QMutexLocker(self._lock):
            self._recording = False
        return self._saved_no

    def set_exposure(self, auto, value):
        """auto=True 自动曝光; 否则手动, value 为 DSHOW 曝光档(-13..0)"""
        if self._cap is None:
            return
        if auto:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def stop(self):
        self._running = False
        self.wait(3000)

    # ---- 采集循环 ----
    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status.emit(f"无法打开相机 {self.cam_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap = cap
        self._running = True
        ok, frame = cap.read()
        if ok:
            self.status.emit(f"相机已打开 {frame.shape[1]}x{frame.shape[0]}")
        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.status.emit("读帧失败")
                time.sleep(0.1)
                continue
            with QtCore.QMutexLocker(self._lock):
                if self._recording:
                    if self._frame_no % self._save_every == 0:
                        path = os.path.join(
                            self._save_dir, f"frame_{self._saved_no:05d}.png"
                        )
                        cv2.imwrite(path, frame)
                        self._saved_no += 1
                    self._frame_no += 1
            self.frame_ready.emit(frame)
        cap.release()
        self._cap = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanDIC 滑台扫描采集")
        self.thread = None
        self.recording = False
        self.last_frame = None
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # 预览区
        self.preview = QtWidgets.QLabel("未打开相机")
        self.preview.setMinimumSize(960, 540)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:#202020; color:#aaa;")
        vbox.addWidget(self.preview, stretch=1)

        # 第一行: 相机设置
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
        row1.addStretch(1)
        vbox.addLayout(row1)

        # 第二行: 曝光
        row2 = QtWidgets.QHBoxLayout()
        self.auto_exp = QtWidgets.QCheckBox("自动曝光")
        self.auto_exp.setChecked(True)
        self.auto_exp.stateChanged.connect(self.apply_exposure)
        row2.addWidget(self.auto_exp)
        row2.addWidget(QtWidgets.QLabel("手动曝光档(-13~0):"))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(-13, 0)
        self.exp_slider.setValue(-6)
        self.exp_slider.valueChanged.connect(self.apply_exposure)
        row2.addWidget(self.exp_slider)
        self.exp_label = QtWidgets.QLabel("-6")
        row2.addWidget(self.exp_label)
        vbox.addLayout(row2)

        # 第三行: 录制
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("每 N 帧存 1 帧:"))
        self.save_every = QtWidgets.QSpinBox()
        self.save_every.setRange(1, 30)
        self.save_every.setValue(1)
        row3.addWidget(self.save_every)
        self.rec_btn = QtWidgets.QPushButton("开始录制")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self.toggle_record)
        row3.addWidget(self.rec_btn)
        self.snap_btn = QtWidgets.QPushButton("拍一张")
        self.snap_btn.setEnabled(False)
        self.snap_btn.clicked.connect(self.snapshot)
        row3.addWidget(self.snap_btn)
        row3.addStretch(1)
        vbox.addLayout(row3)

        # 状态栏
        self.info_label = QtWidgets.QLabel("就绪")
        vbox.addWidget(self.info_label)

        # 预览刷新限频 + 清晰度计算限频
        self._last_preview_t = 0.0
        self._last_sharp_t = 0.0
        self._sharpness = 0.0
        self._fps_count = 0
        self._fps_t = time.time()
        self._fps = 0.0

    # ---- 相机开关 ----
    def toggle_camera(self):
        if self.thread is None:
            idx = int(self.cam_combo.currentText())
            w, h = map(int, self.res_combo.currentText().split("x"))
            self.thread = CameraThread(idx, w, h)
            self.thread.frame_ready.connect(self.on_frame)
            self.thread.status.connect(self.info_label.setText)
            self.thread.start()
            self.open_btn.setText("关闭相机")
            self.rec_btn.setEnabled(True)
            self.snap_btn.setEnabled(True)
            QtCore.QTimer.singleShot(500, self.apply_exposure)
        else:
            if self.recording:
                self.toggle_record()
            self.thread.stop()
            self.thread = None
            self.open_btn.setText("打开相机")
            self.rec_btn.setEnabled(False)
            self.snap_btn.setEnabled(False)
            self.preview.setText("未打开相机")
            self.preview.setPixmap(QtGui.QPixmap())

    def apply_exposure(self):
        self.exp_label.setText(str(self.exp_slider.value()))
        self.exp_slider.setEnabled(not self.auto_exp.isChecked())
        if self.thread is not None:
            self.thread.set_exposure(self.auto_exp.isChecked(), self.exp_slider.value())

    # ---- 录制 ----
    def toggle_record(self):
        if not self.recording:
            run_dir = os.path.join(
                CAPTURE_ROOT, time.strftime("run_%Y%m%d_%H%M%S")
            )
            self.thread.start_recording(run_dir, self.save_every.value())
            self.recording = True
            self.run_dir = run_dir
            self.rec_btn.setText("停止录制")
            self.info_label.setText(f"录制中 -> {run_dir}")
        else:
            n = self.thread.stop_recording()
            self.recording = False
            self.rec_btn.setText("开始录制")
            self.info_label.setText(
                f"已停止, 共保存 {n} 帧到 {self.run_dir}\n"
                f"拼接: python stitch.py \"{self.run_dir}\" -o panorama.png"
            )

    def snapshot(self):
        if self.last_frame is None:
            return
        os.makedirs(CAPTURE_ROOT, exist_ok=True)
        path = os.path.join(CAPTURE_ROOT, time.strftime("snap_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, self.last_frame)
        self.info_label.setText(f"已保存 {path}")

    # ---- 帧回调 ----
    def on_frame(self, frame):
        self.last_frame = frame
        now = time.time()
        self._fps_count += 1
        if now - self._fps_t >= 1.0:
            self._fps = self._fps_count / (now - self._fps_t)
            self._fps_count = 0
            self._fps_t = now
        # 清晰度: 每 0.5s 用中心 1/4 区域算一次 Laplacian 方差
        if now - self._last_sharp_t > 0.5:
            self._last_sharp_t = now
            h, w = frame.shape[:2]
            roi = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            self._sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 预览限制在 ~25fps
        if now - self._last_preview_t < 0.04:
            return
        self._last_preview_t = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)
        rec = "● 录制中" if self.recording else ""
        self.statusBar().showMessage(
            f"{w}x{h}  {self._fps:.1f} fps  清晰度={self._sharpness:.0f} (调焦使其最大)  {rec}"
        )

    def closeEvent(self, ev):
        if self.thread is not None:
            self.thread.stop()
        ev.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 760)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
