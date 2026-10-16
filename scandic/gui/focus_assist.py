# -*- coding: utf-8 -*-
"""
调焦辅助: 实时显示指定相机中心区域的清晰度指标, 拧对焦环使数值最大。

用法:
    python focus_assist.py                    # 默认第一台
    python focus_assist.py BK24404AAK00007    # 指定序列号
按 Ctrl+C 退出。
"""
import sys
import time

import cv2

from scandic.hardware.dahua_cam import DahuaCamera


def main():
    serial = sys.argv[1] if len(sys.argv) > 1 else None
    cam = DahuaCamera.open(serial=serial)
    cam.start()
    best = 0.0
    print("拧动对焦环, 使清晰度最大 (Ctrl+C 退出)")
    try:
        while True:
            img = cam.get_frame(2000)
            if img is None:
                continue
            h, w = img.shape
            roi = img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
            sharp = cv2.Laplacian(roi, cv2.CV_64F).var()
            best = max(best, sharp)
            bar = "#" * min(60, int(sharp / 50))
            print(f"\r清晰度 {sharp:7.0f}  历史最高 {best:7.0f}  亮度 {img.mean():5.1f}  {bar:<60}",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()
