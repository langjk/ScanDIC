# -*- coding: utf-8 -*-
"""
原始帧录制(诊断用): 帧差检测运动, 运动期间保存全部原始帧(不裁剪不配准)。

用法: python record_raw.py [--cam 1] [--max-sec 240] [--still-sec 8]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--max-sec", type=float, default=240)
    ap.add_argument("--still-sec", type=float, default=8)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("错误: 相机打不开(Windows 相机App开着会占用, 请关掉)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    t0 = time.time()
    while time.time() - t0 < 2.0:
        cap.read()
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    if exp is not None and -14 < exp < 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        print(f"曝光已锁定 {exp}", flush=True)

    from scandic.paths import CAPTURES
    run_dir = os.path.join(str(CAPTURES), time.strftime("raw_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"输出目录: {run_dir}", flush=True)

    def small_gray(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=0.125, fy=0.125,
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    # 标定静止时的帧差噪声底
    prev = None
    noise = []
    for _ in range(30):
        ok, f = cap.read()
        if not ok:
            continue
        s = small_gray(f)
        if prev is not None:
            noise.append(np.abs(s - prev).mean())
        prev = s
    floor = float(np.median(noise))
    thresh = max(floor * 4.0, floor + 0.8)
    print(f"帧差噪声底 {floor:.3f}, 运动阈值 {thresh:.3f}", flush=True)
    print(">>> 就绪, 现在推滑台/云台; 推完停住 8 秒自动结束", flush=True)

    n_saved = 0
    moved = False
    last_move_t = time.time()
    t_start = time.time()
    last_report = 0.0
    ts_file = open(os.path.join(run_dir, "timestamps.csv"), "w", encoding="utf-8")
    ts_file.write("frame,index,t_sec,diff\n")

    while True:
        now = time.time()
        if now - t_start > args.max_sec:
            print("达到最长等待时间, 结束", flush=True)
            break
        if moved and now - last_move_t > args.still_sec:
            print("运动已停止, 自动结束", flush=True)
            break
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        s = small_gray(frame)
        diff = float(np.abs(s - prev).mean())
        prev = s
        moving_now = diff > thresh
        if moving_now:
            moved = True
            last_move_t = now
        # 运动中 + 停止后拖尾 1.5s 都存
        if moved and (moving_now or now - last_move_t < 1.5):
            cv2.imwrite(os.path.join(run_dir, f"frame_{n_saved:05d}.png"),
                        frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            ts_file.write(f"frame_{n_saved:05d},{n_saved},{now-t_start:.3f},{diff:.3f}\n")
            n_saved += 1
        if now - last_report > 2.0:
            last_report = now
            print(f"  t={now-t_start:5.1f}s diff={diff:6.3f} "
                  f"{'运动中' if moving_now else '静止'} 已存 {n_saved} 帧", flush=True)

    ts_file.close()
    cap.release()
    print(f"完成: 共保存 {n_saved} 帧到 {run_dir}", flush=True)


if __name__ == "__main__":
    main()
