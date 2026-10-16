# -*- coding: utf-8 -*-
"""
大华(华睿) MV-A7A20MU30 相机封装 —— 基于 MV Viewer 自带 Python SDK。

依赖: mvsdk/MVSDK.py (从 MV Viewer 安装目录拷贝并打了 64 位 DLL 路径补丁)。
Mono8 直接零转换出 numpy 灰度图。

用法:
    from scandic.hardware.dahua_cam import DahuaCamera
    for info in DahuaCamera.enumerate():
        print(info)
    cam = DahuaCamera.open(serial="BK06664AAK00010")
    cam.set_exposure(8000)          # us
    cam.start()
    img = cam.get_frame()           # numpy uint8 (H, W)
    cam.stop(); cam.close()
"""
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mvsdk"))
from MVSDK import *  # noqa: F401,F403  (ctypes 封装, 大量符号)

_camera_list_keeper = []  # 保持 discovery 返回的列表存活
_sdk_lock = threading.Lock()  # SDK 枚举/连接非线程安全, 全局串行化


def _enum_once():
    system = pointer(GENICAM_System())
    if GENICAM_getSystemInstance(byref(system)) != 0:
        raise RuntimeError("getSystemInstance 失败")
    camera_list = pointer(GENICAM_Camera())
    cnt = c_uint()
    if system.contents.discovery(system, byref(camera_list), byref(cnt),
                                 c_int(GENICAM_EProtocolType.typeAll)) != 0:
        raise RuntimeError("相机枚举失败")
    _camera_list_keeper.append(camera_list)
    return [camera_list[i] for i in range(cnt.value)]


def _enum_raw(retries=3, delay=0.5):
    """带重试的枚举: SDK 冷启动首次 discovery 可能失败/返回空。
    调用方需持有 _sdk_lock。"""
    last = None
    for k in range(retries):
        try:
            cams = _enum_once()
            if cams:
                return cams
            last = RuntimeError("没有发现相机")
        except RuntimeError as e:
            last = e
        time.sleep(delay)
    raise last


class DahuaCamera:
    def __init__(self, camera):
        self._cam = camera
        self._stream = None
        self._acq = None  # AcquisitionControl 缓存(软触发用)
        self._soft_trigger = False

    # ---------- 枚举 / 打开 ----------

    @staticmethod
    def enumerate():
        with _sdk_lock:
            cams = _enum_raw()
        out = []
        for c in cams:
            out.append({
                "key": c.getKey(c).decode(errors="replace"),
                "vendor": c.getVendorName(c).decode(errors="replace"),
                "model": c.getModelName(c).decode(errors="replace"),
                "serial": c.getSerialNumber(c).decode(errors="replace"),
            })
        return out

    @classmethod
    def open(cls, serial=None, index=0):
        # 全程持锁: 两个相机线程并发 open 时, SDK 冷启动的 discovery
        # 会互相打架(首次必挂), 串行化 + 重试后一次就能成功
        with _sdk_lock:
            cams = _enum_raw()
            cam = None
            if serial:
                for c in cams:
                    if c.getSerialNumber(c).decode(errors="replace") == serial:
                        cam = c
                        break
                if cam is None:
                    raise RuntimeError(f"找不到序列号 {serial} 的相机")
            else:
                cam = cams[index]
            if cam.connect(cam, c_int(
                    GENICAM_ECameraAccessPermission.accessPermissionControl)) != 0:
                raise RuntimeError(
                    "相机连接失败(若 MV Viewer 软件还开着且占用相机, 请先断开/关闭)")
            return cls(cam)

    def close(self):
        if self._stream is not None:
            self.stop()
        if self._soft_trigger:
            # 恢复连续模式: 触发设置保存在相机里, 不复位会导致
            # MV Viewer 等其它软件取流 0 帧
            try:
                self.set_trigger_off()
                self._soft_trigger = False
            except RuntimeError:
                pass
        if self._acq is not None:
            self._acq.contents.release(self._acq)
            self._acq = None
        self._cam.disConnect(byref(self._cam))

    # ---------- 属性节点 ----------

    def _set_enum(self, name, symbol):
        node = pointer(GENICAM_EnumNode())
        info = GENICAM_EnumNodeInfo()
        info.pCamera = pointer(self._cam)
        info.attrName = name.encode()
        if GENICAM_createEnumNode(byref(info), byref(node)) != 0:
            raise RuntimeError(f"创建节点 {name} 失败")
        ret = node.contents.setValueBySymbol(node, symbol.encode())
        node.contents.release(node)
        if ret != 0:
            raise RuntimeError(f"设置 {name}={symbol} 失败")

    def _set_double(self, name, value):
        node = pointer(GENICAM_DoubleNode())
        info = GENICAM_DoubleNodeInfo()
        info.pCamera = pointer(self._cam)
        info.attrName = name.encode()
        if GENICAM_createDoubleNode(byref(info), byref(node)) != 0:
            raise RuntimeError(f"创建节点 {name} 失败")
        ret = node.contents.setValue(node, c_double(value))
        node.contents.release(node)
        if ret != 0:
            raise RuntimeError(f"设置 {name}={value} 失败")

    def _set_bool(self, name, value):
        node = pointer(GENICAM_BoolNode())
        info = GENICAM_BoolNodeInfo()
        info.pCamera = pointer(self._cam)
        info.attrName = name.encode()
        if GENICAM_createBoolNode(byref(info), byref(node)) != 0:
            raise RuntimeError(f"创建节点 {name} 失败")
        ret = node.contents.setValue(node, c_uint(1 if value else 0))
        node.contents.release(node)
        if ret != 0:
            raise RuntimeError(f"设置 {name}={value} 失败")

    def get_double(self, name):
        node = pointer(GENICAM_DoubleNode())
        info = GENICAM_DoubleNodeInfo()
        info.pCamera = pointer(self._cam)
        info.attrName = name.encode()
        if GENICAM_createDoubleNode(byref(info), byref(node)) != 0:
            raise RuntimeError(f"创建节点 {name} 失败")
        v = c_double()
        ret = node.contents.getValue(node, byref(v))
        node.contents.release(node)
        if ret != 0:
            raise RuntimeError(f"读取 {name} 失败")
        return v.value

    def set_exposure_auto(self, on):
        """on=True 连续自动曝光; False 关闭(保持当前值)。"""
        self._set_enum("ExposureAuto", "Continuous" if on else "Off")

    def set_exposure(self, exposure_us):
        """关自动曝光并设定曝光时间(微秒)。"""
        try:
            self._set_enum("ExposureAuto", "Off")
        except RuntimeError:
            pass  # 部分型号无此节点
        self._set_double("ExposureTime", float(exposure_us))

    def set_trigger_off(self):
        self._set_enum("TriggerMode", "Off")

    def set_frame_rate(self, fps):
        try:
            self._set_bool("AcquisitionFrameRateEnable", True)
        except RuntimeError:
            pass
        self._set_double("AcquisitionFrameRate", float(fps))

    # ---------- 软触发 ----------

    def set_soft_trigger(self):
        """切到软触发模式: 只有 send_trigger() 后才出一帧。"""
        self._set_enum("TriggerSelector", "FrameStart")
        self._set_enum("TriggerMode", "On")
        self._set_enum("TriggerSource", "Software")

    def _acq_ctrl(self):
        if self._acq is None:
            info = GENICAM_AcquisitionControlInfo()
            info.pCamera = pointer(self._cam)
            acq = pointer(GENICAM_AcquisitionControl())
            if GENICAM_createAcquisitionControl(pointer(info), byref(acq)) != 0:
                raise RuntimeError("创建 AcquisitionControl 失败")
            self._acq = acq
        return self._acq

    def send_trigger(self):
        """发一次软触发(需先 set_soft_trigger + start)。"""
        acq = self._acq_ctrl()
        node = acq.contents.triggerSoftware(acq)
        ret = node.execute(byref(node))
        node.release(byref(node))
        if ret != 0:
            raise RuntimeError("软触发执行失败")

    # ---------- 取流 ----------

    def start(self, soft_trigger=False):
        if soft_trigger:
            self.set_soft_trigger()
            self._soft_trigger = True
        else:
            self.set_trigger_off()
            self._soft_trigger = False
        info = GENICAM_StreamSourceInfo()
        info.channelId = 0
        info.pCamera = pointer(self._cam)
        stream = pointer(GENICAM_StreamSource())
        if GENICAM_createStreamSource(pointer(info), byref(stream)) != 0:
            raise RuntimeError("创建流失败")
        if stream.contents.startGrabbing(
                stream, c_ulonglong(0),
                c_int(GENICAM_EGrabStrategy.grabStrartegySequential)) != 0:
            stream.contents.release(stream)
            raise RuntimeError("开始拉流失败")
        self._stream = stream

    def get_frame(self, timeout_ms=1000, with_ts=False):
        """取一帧, 返回 numpy uint8 灰度图(Mono8), 超时/无效返回 None。

        with_ts=True 时返回 (img, 设备时间戳)。
        """
        frame = pointer(GENICAM_Frame())
        if self._stream.contents.getFrame(self._stream, byref(frame),
                                          c_uint(timeout_ms)) != 0:
            return (None, 0) if with_ts else None
        if frame.contents.valid(frame) != 0:
            frame.contents.release(frame)
            return (None, 0) if with_ts else None
        h = frame.contents.getImageHeight(frame)
        w = frame.contents.getImageWidth(frame)
        size = frame.contents.getImageSize(frame)
        pf = frame.contents.getImagePixelFormat(frame)
        ts = frame.contents.getImageTimeStamp(frame)
        buf = frame.contents.getImage(frame)
        data = string_at(c_char_p(buf), size)
        frame.contents.release(frame)
        if pf != EPixelType.gvspPixelMono8:
            raise RuntimeError(f"仅支持 Mono8, 当前像素格式 {pf}")
        img = np.frombuffer(data, np.uint8).reshape(h, w).copy()
        return (img, ts) if with_ts else img

    def stop(self):
        if self._stream is not None:
            self._stream.contents.stopGrabbing(self._stream)
            self._stream.contents.release(self._stream)
            self._stream = None


if __name__ == "__main__":
    for i, info in enumerate(DahuaCamera.enumerate()):
        print(i, info)
