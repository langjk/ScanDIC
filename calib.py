# -*- coding: utf-8 -*-
"""鱼眼去畸变(等距投影模型), 参数来自 undistort.json(实拍位移场拟合)。"""
import json
import os

import cv2
import numpy as np

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "undistort.json")


def load_maps(json_path=DEFAULT_JSON):
    """返回 (map1, map2, (W,H)) 或 None(无标定文件)。"""
    if not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as fp:
        p = json.load(fp)
    W, H = p["width"], p["height"]
    f, fyr = p["f"], p["fyr"]
    K = np.array([[f, 0, p["cx"]], [0, f * fyr, p["cy"]], [0, 0, 1.0]])
    D = np.array([p.get("k1", 0.0), 0, 0, 0.0])
    Knew = np.array([[f * p["out_scale"], 0, W / 2],
                     [0, f * p["out_scale"] * fyr, H / 2], [0, 0, 1.0]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (W, H), cv2.CV_16SC2)
    return m1, m2, (W, H)


def undistort(frame, maps):
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
