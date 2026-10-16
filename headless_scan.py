# -*- coding: utf-8 -*-
"""
无界面自动扫描: 启动后等待滑台移动, 运动停止 8 秒后自动结束并保存全景。

用法: python headless_scan.py [--cam 1] [--max-sec 240] [--crop 0.6]

针对广角镜头: 只取画面中央 crop 比例的区域参与拼接(边角软/畸变大)。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib
from live_stitcher import LiveStitcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--max-sec", type=float, default=240)
    ap.add_argument("--still-sec", type=float, default=8,
                    help="运动开始后静止该秒数则自动结束")
    ap.add_argument("--crop", type=float, default=0.6,
                    help="只用画面中央该比例区域(广角镜头边角软/畸变大)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("错误: 相机打不开(Windows 相机App开着会占用, 请关掉)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    t0 = time.time()
    while time.time() - t0 < 2.0:   # 自动曝光稳定
        cap.read()
    # 尝试锁定曝光(拼接期间亮度恒定)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    locked = False
    if exp is not None and -14 < exp < 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        ok, f = cap.read()
        if ok and f.mean() > 20:
            locked = True
    if not locked:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    print(f"曝光: {'已锁定 ' + str(exp) if locked else '保持自动'}", flush=True)

    maps = calib.load_maps()
    print(f"去畸变: {'开启(undistort.json)' if maps else '无标定文件, 关闭'}", flush=True)

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "captures", time.strftime("scan_%Y%m%d_%H%M%S"))
    st = LiveStitcher(save_dir=run_dir)
    print(f"输出目录: {run_dir}", flush=True)
    print(">>> 就绪, 现在可以推滑台(匀速慢推, 一趟推到头, 停住不动即自动结束)",
          flush=True)

    c = args.crop
    moved = False
    last_move_t = time.time()
    t_start = time.time()
    counts = {}
    last_report = 0.0
    last_info = None
    prev_disp = None

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
            print("读帧失败", flush=True)
            time.sleep(0.2)
            continue
        if maps:
            frame = calib.undistort(frame, maps)
        h, w = frame.shape[:2]
        y0 = int(h * (1 - c) / 2) & ~1
        x0 = int(w * (1 - c) / 2) & ~1
        roi = frame[y0 : y0 + (int(h * c) & ~1), x0 : x0 + (int(w * c) & ~1)]
        info = st.add_frame(np.ascontiguousarray(roi))
        last_info = info
        s = info["state"]
        counts[s] = counts.get(s, 0) + 1
        # 运动判据: 位移相对上一帧仍在变化(推到一半停住时 state 一直是
        # tracking, 不能拿状态当运动依据); 丢失帧一律视为在动
        disp = (info["dx"], info["dy"])
        if s == "lost":
            moving = True
        elif prev_disp is None:
            moving = False
        else:
            moving = (abs(disp[0] - prev_disp[0]) > 0.8
                      or abs(disp[1] - prev_disp[1]) > 0.8)
        prev_disp = disp
        if moving:
            last_move_t = now
            if s in ("tracking", "keyframe"):
                moved = True
        if s == "keyframe":
            print(f"  [关键帧 {info['n_key']}] 位置=({info['pos'][0]:.0f},"
                  f"{info['pos'][1]:.0f}) NCC={info['ncc']:.3f}", flush=True)
        if now - last_report > 2.0:
            last_report = now
            print(f"  t={now-t_start:5.1f}s {s:9s} 关键帧={info['n_key']} "
                  f"位移=({info['dx']:+7.1f},{info['dy']:+6.1f}) "
                  f"NCC={info['ncc']:.3f} 丢失={info['n_lost']}", flush=True)

    cap.release()
    print(f"状态统计: {counts}", flush=True)
    print("融合中...", flush=True)
    pano, pos = st.finish()
    if pano is None:
        sys.exit("没有拼到任何内容(滑台没动过?)")
    small = pano
    if small.shape[1] > 1600:
        sc = 1600 / small.shape[1]
        small = cv2.resize(pano, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(run_dir, "pano_small.png"), small)
    print(f"完成: 关键帧 {len(pos)} 个, 全景 {pano.shape[1]}x{pano.shape[0]}px, "
          f"丢失 {st.n_lost} 次", flush=True)
    print(f"全景: {os.path.join(run_dir, 'panorama.png')}", flush=True)


if __name__ == "__main__":
    main()
