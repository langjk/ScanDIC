# -*- coding: utf-8 -*-
"""
双目标定图像采集(软触发同步对拍)。

用法:
    python calib_capture.py --cols 11 --rows 8            # 棋盘格内角点数
    python calib_capture.py --cols 11 --rows 8 --auto 2   # 每2秒自动拍(双目都检出角点才存)
    python calib_capture.py --exp 8000                    # 顺带设曝光(us)

操作: 回车拍一对, q+回车 结束。每对存为 left_XX.png / right_XX.png。
采集建议: 15~25 对, 标定板占画面 1/3 以上, 覆盖视场各处并带俯仰/偏转姿态。
"""
import argparse
import os
import sys
import time

import cv2

from dahua_cam import DahuaCamera

LEFT_SN = "BK24404AAK00007"   # 左相机序列号(按机位实际情况可调换)
RIGHT_SN = "BK06664AAK00010"


def make_blob_detector(scale=1.0):
    """圆点板斑点检测器, 面积阈值按图像缩放比例调整。"""
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 150 * scale * scale
    p.maxArea = 50000 * scale * scale
    p.filterByCircularity = True
    p.minCircularity = 0.6
    p.filterByConvexity = False
    p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)


def detect_quick(img, pattern, kind):
    """降采样快速检测(仅用于采集时反馈)。"""
    s = 0.35
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if kind == "circles":
        found, _ = cv2.findCirclesGrid(
            small, pattern, flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=make_blob_detector(s))
    else:
        found, _ = cv2.findChessboardCornersSB(
            small, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, required=True, help="横向点数/内角点数")
    ap.add_argument("--rows", type=int, required=True, help="纵向点数/内角点数")
    ap.add_argument("--type", choices=("circles", "chess"), default="circles",
                    help="标定板类型: circles=圆点阵(默认) chess=棋盘格")
    ap.add_argument("--exp", type=float, default=None, help="曝光(us), 不给则用相机当前值")
    ap.add_argument("--auto", type=float, default=None,
                    help="自动模式: 每 N 秒拍一对, 双目均检出角点才保存")
    args = ap.parse_args()
    pattern = (args.cols, args.rows)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures",
                       time.strftime("stereo_calib_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"输出目录: {out}")

    camL = DahuaCamera.open(serial=LEFT_SN)
    camR = DahuaCamera.open(serial=RIGHT_SN)
    for c in (camL, camR):
        if args.exp:
            c.set_exposure(args.exp)
        c.start(soft_trigger=True)

    def grab_pair():
        camL.send_trigger()
        camR.send_trigger()
        return camL.get_frame(2000), camR.get_frame(2000)

    n = 0
    try:
        if args.auto:
            print(f"自动模式: 每 {args.auto}s 尝试一对 (Ctrl+C 结束)。挪动标定板换姿态...")
            while True:
                a, b = grab_pair()
                if a is None or b is None:
                    print("取帧失败, 重试")
                    continue
                fa, fb = detect_quick(a, pattern, args.type), detect_quick(b, pattern, args.type)
                if fa and fb:
                    cv2.imwrite(os.path.join(out, f"left_{n:02d}.png"), a)
                    cv2.imwrite(os.path.join(out, f"right_{n:02d}.png"), b)
                    n += 1
                    print(f"[{n:02d}] 已保存 (L:OK R:OK) —— 请换一个姿态")
                else:
                    print(f"     跳过 (L:{'OK' if fa else '--'} R:{'OK' if fb else '--'})")
                time.sleep(args.auto)
        else:
            while True:
                key = input(f"[已存 {n} 对] 回车拍摄 / q结束 > ").strip().lower()
                if key == "q":
                    break
                a, b = grab_pair()
                if a is None or b is None:
                    print("取帧失败")
                    continue
                fa, fb = detect_quick(a, pattern, args.type), detect_quick(b, pattern, args.type)
                if fa and fb:
                    cv2.imwrite(os.path.join(out, f"left_{n:02d}.png"), a)
                    cv2.imwrite(os.path.join(out, f"right_{n:02d}.png"), b)
                    n += 1
                    print(f"已保存 #{n:02d} (L:OK R:OK)")
                else:
                    print(f"未保存: 角点检测 L:{'OK' if fa else '失败'} "
                          f"R:{'OK' if fb else '失败'} —— 调整标定板位置/朝向/照明")
    except KeyboardInterrupt:
        pass
    finally:
        for c in (camL, camR):
            c.stop()
            c.close()
    print(f"\n共 {n} 对 -> {out}")
    print(f"标定: python stereo_calibrate.py \"{out}\" --type {args.type} "
          f"--cols {args.cols} --rows {args.rows} --square <点中心距mm>")


if __name__ == "__main__":
    main()
