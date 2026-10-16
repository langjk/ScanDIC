# -*- coding: utf-8 -*-
"""把两台相机的 TriggerMode 恢复为 Off(连续出图模式)。

软触发测试/标定采集如果异常退出, 相机会停留在触发模式, MV Viewer 里
表现为 0 FPS 无画面 —— 跑一次本脚本即可恢复。
运行前先在 MV Viewer 里断开相机(或关闭 MV Viewer), 否则连接会被占用。
"""
from dahua_cam import DahuaCamera

for info in DahuaCamera.enumerate():
    sn = info["serial"]
    try:
        cam = DahuaCamera.open(serial=sn)
        cam.set_trigger_off()
        cam.close()
        print(f"{sn}: TriggerMode 已恢复 Off")
    except RuntimeError as e:
        print(f"{sn}: 失败 - {e}")
