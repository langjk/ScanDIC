# -*- coding: utf-8 -*-
"""
扫描式 DIC 采集: 电机驱动滑台逐站位停稳, 双目软触发多帧平均采集。

用法:
    python scan_capture.py ref --seek-edge --auto-edge
    python scan_capture.py def --plan captures/scan_ref_xxxxxx

ref: 基准扫描 —— 开头自动用小位移探测"像素/脉冲"比例, 按目标重叠率
     规划站距, 逐站采集后电机返回起点。站位表存 meta.json。
def: 变形后复扫 —— 图像起点对齐后复现 ref 站位表。回位误差必须保留，
     只有独立静止参照才能分离相机运动；重叠一致性不保证绝对准确度。

约定: 默认不限制累计脉冲（用户2026-09-20更新）；正常结束回板边原点。
运动分段并持续查堵转/限位，故障或中断立即停止，不在位置不明时盲目回程。
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scandic.processing.stitch import register_pair, highpass

from scandic.paths import PROJECT_ROOT as ROOT
# 相机序列号与左右指派以 calib_gui 为唯一来源: 标定就是用它那套 L/R
# 指派做的, 扫描必须完全一致(否则 R,T 用反/认错机); 避免各处常量漂移。
from scandic.gui.calib_gui import LEFT_SN, RIGHT_SN  # noqa: E402
SETTLE_S = 1.0     # 到位后静置时间(长焦对振动敏感)
RPM, ACC = 300, 120
MAX_SEGMENT_PULSES = 8000
FAULT_KEYS = ("stalled", "stall_protect", "limit_left", "limit_right")


def newest_calib():
    files = sorted(glob.glob(os.path.join(
        ROOT, "captures", "stereo_calib_*", "stereo_calib.npz")))
    if not files:
        raise RuntimeError("未找到双目标定")
    return files[-1]


class ScanLog:
    """每个阶段同步元数据、扫描日志和项目工作记录。"""
    def __init__(self, out, meta, resume=False):
        self.out, self.meta = out, meta
        os.makedirs(out, exist_ok=resume)

    def event(self, message, **updates):
        self.meta.update(updates)
        self.meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = os.path.join(self.out, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(self.meta, fp, ensure_ascii=False, indent=2, allow_nan=False)
        # Windows索引器/文件查看器可能短暂占用目标；保留原子替换并有限重试。
        for attempt in range(8):
            try:
                os.replace(tmp, os.path.join(self.out, "meta.json"))
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.025 * 2 ** attempt, 0.4))
        line = f"- {self.meta['updated']} [{os.path.basename(self.out)}] {message}\n"
        print(message, flush=True)
        for path in (os.path.join(self.out, "scan_log.md"), os.path.join(ROOT, "WORK_LOG.md")):
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(line)


class Rig:
    def __init__(self, budget=None, exposure_us=None):
        if budget is not None and budget <= 0:
            raise ValueError("可选 budget 必须为正数")
        self.budget = budget
        self.pos = self.origin = self.min_pos = self.max_pos = 0
        self.faulted = False
        self.camL = self.camR = self.motor = None
        # 延迟导入：CLI 参数和模拟测试不加载相机 DLL、不打开硬件。
        from scandic.hardware.dahua_cam import DahuaCamera
        from scandic.hardware.zdt_motor import ZdtMotor
        try:
            self.motor = ZdtMotor("COM4")
            self.check_status(require_enabled=False)
            self.camL = DahuaCamera.open(serial=LEFT_SN)
            self.camR = DahuaCamera.open(serial=RIGHT_SN)
            for c in (self.camL, self.camR):
                if exposure_us is None:
                    c.set_exposure_auto(False)
                else:
                    c.set_exposure(exposure_us)
                c.start(soft_trigger=True)
            self.exposures = [c.get_double("ExposureTime") for c in (self.camL, self.camR)]
            if not self.motor.read_status()["enabled"]:
                raise RuntimeError("电机未使能，请现场确认后使能")
            # 中断/手动挪动后上一目标可能失效；停止并确认编码器稳定。
            self.motor.stop()
            first_deg = self.motor.read_position_deg()
            time.sleep(0.25)
            self.start_deg = self.motor.read_position_deg()
            if abs(self.start_deg - first_deg) > 0.2:
                raise RuntimeError("初始化时电机位置不稳定")
        except BaseException:
            self.close(home=False)
            raise

    def check_status(self, require_enabled=True):
        st = self.motor.read_status()
        if any(st[k] for k in FAULT_KEYS) or (require_enabled and not st["enabled"]):
            raise RuntimeError(f"电机异常: {st}")
        return st

    def stop(self):
        if self.motor is not None:
            try:
                self.motor.stop()
            except Exception as exc:
                print(f"停止命令失败: {exc}", flush=True)

    def close(self, home=False):
        try:
            if home:
                if self.faulted:
                    raise RuntimeError("位置可能不明，禁止故障后自动回原点")
                self.goto_station(0)
                self.check_status()
        finally:
            for c in (self.camL, self.camR):
                if c is not None:
                    try:
                        c.close()
                    except Exception as exc:
                        print(f"关闭相机失败: {exc}", flush=True)
            if self.motor is not None:
                self.motor.close()
            self.camL = self.camR = self.motor = None

    def goto_station(self, station_offset):
        self.goto_abs(self.origin + int(station_offset))

    def check_target(self, pulse_pos):
        target = int(pulse_pos)
        lo, hi = min(self.min_pos, target), max(self.max_pos, target)
        if self.budget is not None and hi - lo > self.budget:
            raise RuntimeError(f"运动拒绝: 位置跨度 {hi-lo} 超过预算 {self.budget} 脉冲")
        if self.faulted:
            raise RuntimeError("故障状态禁止继续运动")
        return target, lo, hi

    def goto_abs(self, pulse_pos):
        target, lo, hi = self.check_target(pulse_pos)
        d = target - self.pos
        try:
            self.check_status()
            # 从程序启动时编码器位置验收；不靠电机绝对位置保存扫描原点。
            target_deg = self.start_deg + target * 360.0 / 3200.0
            current_deg = self.motor.read_position_deg()
            if abs(current_deg - (self.start_deg + self.pos * 360.0 / 3200.0)) > 3.6:
                raise RuntimeError("编码器偏离已知位置，禁止继续运动")
            if d == 0:
                return
            # mode=2 从实时位置起算，避免中断前的旧目标引入未知位移。
            pulses = int(round((target_deg - current_deg) * 3200.0 / 360.0))
            self.motor.move_pulses(pulses, rpm=RPM, acc=ACC, mode=2)
            self.motor.wait_arrival(timeout=120, target_deg=target_deg)
            self.check_status()
            self.pos = target
            self.min_pos, self.max_pos = lo, hi
        except BaseException:
            self.faulted = True
            self.stop()
            raise
        time.sleep(SETTLE_S)

    def pair_avg(self, n):
        if n < 1:
            raise ValueError("平均帧数必须 >=1")
        accL = accR = None
        for _ in range(n):
            self.camL.send_trigger()
            self.camR.send_trigger()
            a = self.camL.get_frame(2000)
            b = self.camR.get_frame(2000)
            if a is None or b is None:
                raise RuntimeError("取帧失败")
            accL = a.astype(np.float32) if accL is None else accL + a
            accR = b.astype(np.float32) if accR is None else accR + b
        return accL / n, accR / n


def save_pair(d, idx, a, b, prefix="station"):
    for img, tag in ((a, "L"), (b, "R")):
        u16 = np.clip(img * 256.0, 0, 65535).astype(np.uint16)
        if not cv2.imwrite(os.path.join(d, f"{prefix}_{idx:02d}_{tag}.png"), u16,
                           [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError("站位图像保存失败")


def texture_frac(img, thresh=0.10):
    """相对对比度覆盖率；亮度变化不应被当作板边，另设噪声底。"""
    s = 0.25
    f = cv2.resize(img.astype(np.float32), None, fx=s, fy=s,
                   interpolation=cv2.INTER_AREA)
    mu = cv2.blur(f, (15, 15))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (15, 15)) - mu * mu, 0))
    return float(((std > 1.5) & (std / (mu + 3.0) > thresh)).mean())


def projective_registration(imgA, imgB):
    """SIFT单应补偿跨站透视/缩放，仅用于粗定位及质量核验。"""
    from scandic.dic_engine import contrast_normalize
    s = 0.5
    a, b = [cv2.resize(im.astype(np.float32), None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA) for im in (imgA, imgB)]
    sift = cv2.SIFT_create(nfeatures=5000)
    ka, da = sift.detectAndCompute(contrast_normalize(a).astype(np.uint8), None)
    kb, db = sift.detectAndCompute(contrast_normalize(b).astype(np.uint8), None)
    if da is None or db is None or min(len(ka), len(kb)) < 30:
        raise RuntimeError("跨站SIFT特征不足")
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < 0.70 * n.distance]
    if len(good) < 30:
        raise RuntimeError("跨站SIFT匹配不足")
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 1.5)
    if H is None or not np.isfinite(H).all() or inl.sum() < 30:
        raise RuntimeError("跨站单应估计失败")
    warped = cv2.warpPerspective(b, H, a.shape[::-1])
    mask = cv2.warpPerspective(np.ones(b.shape, np.uint8), H, a.shape[::-1])
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8)).astype(bool)
    if mask.mean() < 0.15:
        raise RuntimeError("跨站有效重叠不足15%")
    x = contrast_normalize(a)[mask].astype(np.float64)
    y = contrast_normalize(warped)[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    ncc = float(np.dot(x, y) / denom) if denom > 0 else 0.0
    S = np.diag([s, s, 1.0])
    return np.linalg.inv(S) @ H @ S, ncc, int(inl.sum())


def shift_px(imgA, imgB, precise=False):
    """B相对A的视场位移；单应时在图像中心求值，与纹理运动反号。"""
    s = 0.25
    a = cv2.resize(imgA.astype(np.float32), None, fx=s, fy=s,
                   interpolation=cv2.INTER_AREA)
    b = cv2.resize(imgB.astype(np.float32), None, fx=s, fy=s,
                   interpolation=cv2.INTER_AREA)
    dx, dy, ncc = register_pair(highpass(a), highpass(b))
    if not np.isfinite([dx, dy, ncc]).all() or ncc < 0.4:
        H, ncc, _ = projective_registration(imgA, imgB)
        center = np.array([imgB.shape[1] / 2, imgB.shape[0] / 2, 1.0])
        p = H @ center
        dx, dy = p[:2] / p[2] - center[:2]
        coarse=(float(dx),float(dy),ncc)
    else:
        coarse=(dx/s,dy/s,ncc)
    if not precise:
        return coarse
    # The 0.3px origin gate needs full-resolution refinement: quarter-scale
    # phase-correlation centroid bias is too large to use as its final estimate.
    from scandic.processing.precise_match import SplineImage,PreciseMatcher
    a,b=SplineImage(imgA),SplineImage(imgB)
    matcher=PreciseMatcher(half=32,search=16,min_ncc=.7)
    h,w=imgA.shape
    values=[]
    for y in np.linspace(52,h-53,5):
        for x in np.linspace(52,w-53,5):
            m=matcher.match(a,b,x,y,init=(-coarse[0],-coarse[1]))
            if m is not None:
                values.append(m)
    if len(values)<9:
        raise RuntimeError('原点精确对齐有效子区不足9个')
    dx,dy,ncc=np.median(values,axis=0)
    return -float(dx),-float(dy),float(ncc)


def geometry_report(a, b, calib):
    """用与标定 F 无关的 SIFT/单应内点核验全分辨率极线误差。"""
    from scandic.dic_engine import DicEngine
    with np.load(calib) as c:
        size = tuple(map(int, c["image_size"]))
    if a.shape[::-1] != size or b.shape != a.shape:
        raise RuntimeError("采集尺寸与标定不符")
    eng = DicEngine(calib, out_size=size)
    l, r = eng.undistort_pair(a, b)
    _, ni, nm, pl, pr = eng._bootstrap_H(eng._prep(l), eng._prep(r))
    err = eng._epi_err(pl[:, 0], pl[:, 1], pr[:, 0], pr[:, 1])
    report = {"sift_inliers": ni, "sift_matches": nm,
              "epipolar_median_full_px": float(np.median(err)),
              "epipolar_p95_full_px": float(np.percentile(err, 95))}
    if report["epipolar_median_full_px"] > 0.8:
        # SIFT选点受场景/亮度影响；超线先在固定网格上以独立子区定位复核。
        # 匹配与ECC均不使用F约束，不允许靠贴向极线伪造通过。
        from scandic.diagnostics.diagnose_epipolar import diagnose_pair
        verification, _, _ = diagnose_pair(eng, a, b, refine_ecc=True)
        report["subset_verification"] = verification
    return report


def require_geometry(report):
    valid_sift = report["sift_inliers"] >= 30 and np.isfinite(report["epipolar_median_full_px"])
    if valid_sift and report["epipolar_median_full_px"] <= 0.8:
        return
    v = report.get("subset_verification", {})
    if (valid_sift and v.get("ecc_count", 0) >= 100 and
            np.isfinite([v.get("ecc_median_abs_px", np.inf), v.get("ecc_p95_abs_px", np.inf)]).all() and
            v["ecc_median_abs_px"] <= 0.8 and v["ecc_p95_abs_px"] <= 1.5):
        return
    raise RuntimeError(f"当前图像与标定不符: {report}")


def require_shift(dx, dy, ncc, minimum=0.4):
    if not np.isfinite([dx, dy, ncc]).all() or ncc < minimum:
        raise RuntimeError(f"图像配准失败: dx={dx}, dy={dy}, NCC={ncc}")


def motion(rig, log, target, label):
    rig.check_target(target)  # 仅用户显式设置预算时检查总跨度
    log.event(f"{label}: {rig.pos} -> {target} 脉冲",
              pending_target_pulses=target, last_known_position_pulses=rig.pos)
    while rig.pos != target:
        step = max(-MAX_SEGMENT_PULSES, min(MAX_SEGMENT_PULSES, target - rig.pos))
        rig.goto_abs(rig.pos + step)
        log.event(f"{label}: 分段到位 {rig.pos} 脉冲",
                  last_known_position_pulses=rig.pos,
                  visited_range_pulses=[rig.min_pos, rig.max_pos])
    rig.check_status()
    log.event(f"{label}: 已到位 {rig.pos} 脉冲",
              pending_target_pulses=None, last_known_position_pulses=rig.pos,
              visited_range_pulses=[rig.min_pos, rig.max_pos])


def align_origin(rig, log, ref0, vector, tolerance_px=0.3, roi=None):
    vx, vy = map(float, vector)
    v2 = vx * vx + vy * vy
    if not np.isfinite(v2) or v2 <= 0:
        raise ValueError("基准 dir_pxpp 无效")
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("图像对齐容差必须为正有限数")
    if roi is not None:
        x,y,w,h=roi
        if min(x,y)<0 or min(w,h)<128 or x+w>ref0.shape[1] or y+h>ref0.shape[0]:
            raise ValueError("静止参照ROI越界或小于128px")
        ref_compare=ref0[y:y+h,x:x+w]
    else:
        ref_compare=ref0
    for it in range(5):  # 最多4次修正，第5次仅验收
        cur, _ = rig.pair_avg(4)
        current_compare=cur[y:y+h,x:x+w] if roi is not None else cur
        dx, dy, ncc = shift_px(ref_compare, current_compare, precise=True)
        require_shift(dx, dy, ncc)
        off = float(np.hypot(dx, dy))
        log.event(f"原点对齐 {it}: ({dx:+.2f},{dy:+.2f})px NCC={ncc:.3f}")
        if off < tolerance_px:
            rig.origin = rig.pos
            return
        perpendicular = abs(dx * vy - dy * vx) / np.sqrt(v2)
        if perpendicular >= tolerance_px:
            raise RuntimeError(f"横向图像偏差{perpendicular:.3f}px无法用单轴滑台修正")
        if it == 4:
            break
        pulses = int(round(-(dx * vx + dy * vy) / v2))
        if pulses == 0:
            raise RuntimeError("原点误差无法用单轴滑台修正")
        motion(rig, log, rig.pos + pulses, "原点修正")
    raise RuntimeError(f"原点对齐未收敛到{tolerance_px:g}px，禁止复扫")


def capture_station(rig, log, offset, avg, calib, prev=None):
    k = len(log.meta["stations"])
    a, b = rig.pair_avg(avg)
    # 闸门前保留候选原图，失败也能诊断；通过后再提升为正式站位。
    save_pair(log.out, k, a, b, prefix="candidate")
    log.event(f"站位 {k} 候选图已保存，开始质量核验", candidate_station=k)
    tf = [texture_frac(a), texture_frac(b)]
    shift = (0.0, 0.0, 1.0) if prev is None else shift_px(prev, a)
    require_shift(*shift)
    geometry = geometry_report(a, b, calib)
    require_geometry(geometry)
    for tag in ("L", "R"):
        os.replace(os.path.join(log.out, f"candidate_{k:02d}_{tag}.png"),
                   os.path.join(log.out, f"station_{k:02d}_{tag}.png"))
    record = {"index": k, "offset_pulses": offset, "position_pulses": rig.pos,
              "texture_fraction": tf, "shift_px": list(shift[:2]), "ncc": shift[2],
              "geometry": geometry, "encoder_deg": rig.motor.read_position_deg()}
    log.meta["stations"].append(offset)
    log.meta["measured_shift_px"].append(list(shift[:2]))
    log.meta["station_records"].append(record)
    extra = (f"，子区ECC复核中位 {geometry['subset_verification']['ecc_median_abs_px']:.3f}px"
             if 'subset_verification' in geometry else '')
    log.event(f"站位 {k} 已保存 @{offset} 脉冲；纹理 {tf[0]:.1%}/{tf[1]:.1%}，"
              f"相邻 NCC={shift[2]:.3f}，极线中位 {geometry['epipolar_median_full_px']:.3f}px",
              candidate_station=None, last_geometry_confirmation=extra or 'SIFT passed')
    return a, b


def run_reference(args, rig, log, calib):
    a0, b0 = rig.pair_avg(args.avg)
    geometry = geometry_report(a0, b0, calib)
    log.event(f"运动前几何核验: {geometry}", preflight_geometry=geometry)
    require_geometry(geometry)
    if args.seek_edge and texture_frac(a0) < 0.80:
        raise RuntimeError("初始纹理覆盖不足80%，不能可靠寻边；请置于板内并检查照明")
    motion(rig, log, args.probe, "小位移探测")
    ap, _ = rig.pair_avg(1)
    dx, dy, ncc = shift_px(a0, ap)
    require_shift(dx, dy, ncc)
    mag = float(np.hypot(dx, dy))
    if mag < 5:
        raise RuntimeError("探测位移不足5px")
    vector = (dx / args.probe, dy / args.probe)
    ppp = mag / args.probe
    # 沿运动方向换算视场比例，兼容非水平滑台。
    fraction_per_pulse = max(abs(vector[0]) / a0.shape[1],
                             abs(vector[1]) / a0.shape[0])
    step = max(1, int(round((1 - args.overlap) / fraction_per_pulse)))
    back = max(1, int(round(0.30 / fraction_per_pulse)))
    log.event(f"探测: ({dx:+.2f},{dy:+.2f})px，{ppp:.5f}px/脉冲，站距 {step}",
              px_per_pulse=ppp, dir_pxpp=list(vector), step_pulses=step)
    motion(rig, log, 0, "返回探测起点")
    if args.seek_edge:
        while True:
            prior = rig.pos
            motion(rig, log, prior - back, "反向寻板边")
            a, b = rig.pair_avg(1)
            tf = texture_frac(a)
            log.event(f"寻边位置 {rig.pos}：纹理覆盖 {tf:.1%}")
            if tf < 0.80:
                save_pair(log.out, -1, a, b)  # 诊断图，不属于站位表
                log.event("检测到起始板边", start_edge_position_pulses=rig.pos)
                motion(rig, log, prior, "退回板内原点")
                break
    rig.origin = rig.pos
    log.event(f"扫描原点 = {rig.origin} 脉冲", origin_pulses=rig.origin)
    first, _ = capture_station(rig, log, 0, args.avg, calib)
    prev = first
    limit = args.stations if args.stations is not None else (None if args.auto_edge else 5)
    k = 1
    while limit is None or k < limit:
        offset = k * step
        motion(rig, log, rig.origin + offset, f"前往站位 {k}")
        if args.auto_edge:
            a, b = rig.pair_avg(1)
            tf = texture_frac(a)
            if tf < args.edge_frac:
                save_pair(log.out, -2, a, b)  # 尾端诊断图，不进入复扫计划
                log.event(f"另一侧板边触发: 覆盖 {tf:.1%} < {args.edge_frac:.1%}",
                          termination_reason="board_edge", end_edge_position_pulses=rig.pos,
                          end_edge_texture_fraction=tf)
                return first
        prev, _ = capture_station(rig, log, offset, args.avg, calib, prev)
        k += 1
    if args.auto_edge:
        raise RuntimeError("达到站数上限但未检测到另一侧板边，本次扫描不完整")
    log.event("已采完指定站位", termination_reason="requested_stations")
    return first


def validate_plan(meta):
    st = meta.get("stations", [])
    if meta.get("role") != "ref" or meta.get("status") != "completed" or not meta.get("returned_to_origin"):
        raise ValueError("基准扫描未完成或未确认回原点，禁止复扫")
    if (not st or st[0] != 0 or any(type(p) is not int or p < 0 for p in st) or
            any(b <= a for a, b in zip(st, st[1:]))):
        raise ValueError("基准站位表无效")
    v = meta.get("dir_pxpp", [])
    if len(v) != 2 or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("基准运动方向无效")
    if type(meta.get("avg")) is not int or meta["avg"] < 1:
        raise ValueError("基准平均帧数无效")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("ref", "def"))
    ap.add_argument("--stations", type=int, help="站数上限；auto-edge 默认由板边结束，普通扫描默认5")
    ap.add_argument("--auto-edge", action="store_true")
    ap.add_argument("--seek-edge", action="store_true")
    ap.add_argument("--edge-frac", type=float, default=0.55)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--budget", type=int, default=None, help="可选访问跨度限制；默认不限累计脉冲")
    ap.add_argument("--probe", type=int, default=2000)
    ap.add_argument("--avg", type=int, default=4)
    ap.add_argument("--exposure-us", type=float, help="双目固定曝光；省略则保留各相机当前曝光并关闭自动曝光")
    ap.add_argument("--calib", help="ref 使用的标定；默认最新，复制到扫描目录绑定")
    ap.add_argument("--plan", help="def 所用已完成的 ref 目录")
    ap.add_argument("--alignment-tolerance-px", type=float, default=0.3,
                    help="复扫起点图像对齐容差；不代表测量准确度，默认0.3px")
    ap.add_argument("--alignment-roi", nargs=4, type=int, metavar=("X","Y","W","H"),
                    help="原始左图已知静止参照ROI；加载试验不要以变形区做回位基准")
    args = ap.parse_args(argv)
    if not np.isfinite(args.alignment_tolerance_px) or args.alignment_tolerance_px <= 0:
        ap.error("alignment-tolerance-px 必须为正有限数")
    if args.probe <= 0 or (args.budget is not None and
                           (args.budget <= 0 or args.probe > args.budget)):
        ap.error("probe 必须为正；若指定budget，则必须为正且不少于probe")
    if not 0 < args.overlap < 1 or not 0 < args.edge_frac < 0.8:
        ap.error("overlap 必须在(0,1)，edge-frac 必须在(0,0.8)")
    if args.avg < 1 or (args.stations is not None and args.stations < 1):
        ap.error("avg/stations 必须为正整数")
    if args.exposure_us is not None and (not np.isfinite(args.exposure_us) or args.exposure_us <= 0):
        ap.error("exposure-us 必须为正有限数")
    if args.mode == "def" and (not args.plan or args.seek_edge or args.auto_edge or args.calib):
        ap.error("def 必须提供 --plan，且不能使用寻边或另选标定")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = None
    if args.mode == "def":
        with open(os.path.join(args.plan, "meta.json"), encoding="utf-8") as fp:
            plan = json.load(fp)
        validate_plan(plan)
        if args.budget is not None and plan["stations"][-1] > args.budget:
            raise ValueError("基准计划超过本次预算")
        calib_source = plan["calib"]
        for k in range(len(plan["stations"])):
            for tag in ("L", "R"):
                if not os.path.isfile(os.path.join(args.plan, f"station_{k:02d}_{tag}.png")):
                    raise ValueError(f"基准站位 {k} 图像缺失")
    else:
        calib_source = args.calib or newest_calib()
    # 文件有效性先于打开硬件。
    with np.load(calib_source) as c:
        for key in ("K1", "K2", "D1", "D2", "R", "T", "image_size"):
            if key not in c or not np.isfinite(c[key]).all():
                raise ValueError(f"标定无效: {key}")
    out = os.path.join(ROOT, "captures", time.strftime(f"scan_{args.mode}_%Y%m%d_%H%M%S"))
    meta = {"schema_version": 3, "role": args.mode, "status": "initializing",
            "texture_metric": "std>1.5 and std/(mean+3)>0.10 at quarter resolution",
            "shift_method": "translation with projective fallback (image center)",
            "stations": [], "station_records": [], "measured_shift_px": [],
            "avg": args.avg if plan is None else plan["avg"], "budget_pulses": args.budget,
            "returned_to_origin": False, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "options": vars(args), "source_calib": os.path.abspath(calib_source)}
    if plan is not None:
        meta["ref_dir"] = os.path.abspath(args.plan)
        meta["planned_stations"] = plan["stations"]
    log = ScanLog(out, meta)
    calib = os.path.join(out, "stereo_calib.npz")
    shutil.copy2(calib_source, calib)
    log.event(f"开始 {args.mode} 扫描: {out}", calib=calib)
    rig = None
    try:
        rig = Rig(budget=args.budget, exposure_us=args.exposure_us)
        log.event("硬件连接完成，曝光已锁定", status="running", exposure_us=rig.exposures,
                  start_encoder_deg=rig.start_deg)
        if plan is None:
            first = run_reference(args, rig, log, calib)
        else:
            ref0 = cv2.imread(os.path.join(args.plan, "station_00_L.png"), cv2.IMREAD_UNCHANGED)
            if ref0 is None:
                raise ValueError("基准原点图像损坏")
            ref0 = ref0.astype(np.float32) / (256.0 if ref0.dtype == np.uint16 else 1.0)
            a, b = rig.pair_avg(1)
            geo = geometry_report(a, b, calib)
            log.event(f"复扫运动前几何核验: {geo}", preflight_geometry=geo)
            require_geometry(geo)
            align_origin(rig, log, ref0, plan["dir_pxpp"], args.alignment_tolerance_px, args.alignment_roi)
            log.event("复扫原点对齐通过", origin_pulses=rig.origin)
            first = prev = None
            for k, offset in enumerate(plan["stations"]):
                motion(rig, log, rig.origin + offset, f"复扫站位 {k}")
                prev, _ = capture_station(rig, log, offset, plan["avg"], calib, prev)
                if first is None:
                    first = prev
            log.event("基准站位已全部复现", termination_reason="plan_completed")
        motion(rig, log, rig.origin, "正常回原点")
        cur, _ = rig.pair_avg(1)
        dx, dy, ncc = shift_px(first, cur)
        require_shift(dx, dy, ncc)
        if np.hypot(dx, dy) >= 30:
            raise RuntimeError(f"回原点图像验收失败: ({dx:.2f},{dy:.2f})px")
        log.event(f"扫描完成，已回原点；图像残差({dx:+.2f},{dy:+.2f})px NCC={ncc:.3f}",
                  status="completed", returned_to_origin=True,
                  return_check={"shift_px": [dx, dy], "ncc": ncc,
                                "encoder_deg": rig.motor.read_position_deg(),
                                "motor_status": rig.check_status()})
    except BaseException as exc:
        if rig is not None:
            rig.stop()
        log.event(f"扫描中止，不自动回程: {type(exc).__name__}: {exc}",
                  status="aborted", error=f"{type(exc).__name__}: {exc}",
                  position_uncertain=bool(rig is not None and rig.faulted))
        raise
    finally:
        if rig is not None:
            rig.close(home=False)
    return out


if __name__ == "__main__":
    main()
