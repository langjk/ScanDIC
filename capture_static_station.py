"""从已知扫描原点移到指定站，然后保持电机使能静止采集；结束停留目标站。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scan_capture import (Rig, ScanLog, motion, save_pair, geometry_report,
                          require_geometry, shift_px, require_shift)
from scan_pipeline import load_station

ROOT = Path(__file__).resolve().parent


def validate_source(meta, station):
    if (meta.get('status') != 'completed' or not meta.get('returned_to_origin') or
            meta.get('position_uncertain', False) or meta.get('pending_target_pulses') is not None or
            meta.get('last_known_position_pulses') != meta.get('origin_pulses')):
        raise ValueError('源扫描必须完成并停在已知原点')
    plan = meta.get('stations', [])
    if (not 0 <= station < len(plan) or not plan or plan[0] != 0 or
            any(b <= a for a,b in zip(plan,plan[1:]))):
        raise ValueError('目标站或源计划无效')
    return meta['origin_pulses'] + plan[station]


def check_images(old_pair, new_pair, max_shift=30):
    result = [shift_px(a,b) for a,b in zip(old_pair,new_pair)]
    for dx,dy,ncc in result:
        require_shift(dx,dy,ncc,minimum=0.8)
        if np.hypot(dx,dy) >= max_shift:
            raise RuntimeError('图像与保存站位不符，停止实验')
    return result


def geometry_passed(report):
    """诊断采集保留原始帧；原计量闸门判定不变。"""
    try:
        require_geometry(report)
        return True
    except RuntimeError:
        return False


def collect_static(rig, log, states, avg, interval):
    """此函数只读状态/触发相机；不发运动命令，不按图像位移修正。"""
    start = time.monotonic()
    stable_deg = rig.motor.read_position_deg()
    first = last = None
    for i in range(states):
        delay = start + i*interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        status = rig.check_status()
        before = rig.motor.read_position_deg()
        if abs(before-stable_deg) > 0.2:
            raise RuntimeError('静止阶段编码器变化超过0.2度')
        timestamp = time.time()
        last = rig.pair_avg(avg)
        after = rig.motor.read_position_deg()
        if abs(after-stable_deg) > 0.2:
            raise RuntimeError('静止采集期间编码器变化超过0.2度')
        save_pair(log.out, i, *last)
        if first is None:
            first = last
        record = dict(index=i, timestamp_unix=timestamp,
                      elapsed_s=time.monotonic()-start,
                      encoder_before_deg=before, encoder_after_deg=after,
                      motor_status=status)
        log.meta['states'].append(record)
        log.event(f'静止采集{i+1}/{states}，编码器{after:.6f}°',
                  completed_states=i+1)
    return first,last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('--station',type=int,default=11)
    ap.add_argument('--states',type=int,default=21)
    ap.add_argument('--interval',type=float,default=3)
    ap.add_argument('--settle',type=float,default=10)
    args=ap.parse_args(argv)
    if args.states < 2 or not np.isfinite([args.interval,args.settle]).all() or min(args.interval,args.settle)<0:
        ap.error('至少2个状态，间隔/稳定时间须为有限非负数')
    cv2.setNumThreads(4)
    source=args.source.resolve()
    meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    target=validate_source(meta,args.station)
    refs={k:[load_station(source,k,tag) for tag in 'LR'] for k in (0,args.station)}
    with np.load(meta['calib']) as c:
        for name in ('K1','K2','D1','D2','R','T','image_size'):
            if name not in c or not np.isfinite(c[name]).all():
                raise ValueError('源标定损坏')
    out=ROOT/'captures'/time.strftime(f'static_station{args.station:02d}_%Y%m%d_%H%M%S')
    log=ScanLog(str(out),dict(role='static_station',status='initializing',source_scan=str(source),
                            target_station=args.station,target_pulses=target,states=[],avg=meta['avg'],
                            planned_states=args.states,interval_s=args.interval,settle_s=args.settle,
                            no_motion_during_series=True,end_policy='hold_at_target_station',
                            budget_pulses=None,returned_to_origin=False))
    calib=out/'stereo_calib.npz'
    shutil.copy2(meta['calib'],calib)
    log.event('固定站重复采集准备：站'+str(args.station),calib=str(calib))
    rig=None
    try:
        rig=Rig(budget=None)
        expected=meta['return_check']['encoder_deg']
        if abs(rig.start_deg-expected)>3.6:
            raise RuntimeError('当前位置编码器与源扫描回位记录不符，禁止运动')
        if not np.allclose(rig.exposures,meta['exposure_us'],atol=1,rtol=0):
            raise RuntimeError('曝光与扫描数据不符，禁止运动')
        before=rig.pair_avg(meta['avg'])
        check=check_images(refs[0],before)
        geo=geometry_report(*before,str(calib))
        log.event('原点编码器、图像与曝光核对，检查当前几何',origin_image_check=check,preflight_geometry=geo)
        log.event('原点几何检查仅作诊断记录；运动许可由已知位置、图像定位和电机状态核验',
                  preflight_geometry_passed=geometry_passed(geo))
        rig.start_deg=meta['start_encoder_deg']
        rig.pos=rig.origin=meta['origin_pulses']
        rig.min_pos=rig.max_pos=rig.pos
        log.event('开始移至匹配质量较好的站位',status='moving',start_encoder_deg=rig.start_deg,
                  origin_pulses=rig.origin,exposure_us=rig.exposures)
        motion(rig,log,target,f'移动至站{args.station}')
        log.event(f'已到站{args.station}，等待{args.settle:g}秒稳定',status='settling')
        time.sleep(args.settle)
        arrival=rig.pair_avg(meta['avg'])
        save_pair(str(out),0,*arrival,prefix='arrival')
        check=check_images(refs[args.station],arrival)
        geo=geometry_report(*arrival,str(calib))
        log.event('目标站图像及几何核验',arrival_image_check=check,arrival_geometry=geo)
        log.event('目标站几何判定已记录，静止原图完整保存',
                  arrival_geometry_passed=geometry_passed(geo),
                  measurement_quality='geometry_passed' if geometry_passed(geo) else 'diagnostic_only_geometry_gate_failed')
        log.event('静止采集开始，后续不发运动命令',status='acquiring',
                  acquisition_encoder_deg=rig.motor.read_position_deg())
        first,last=collect_static(rig,log,args.states,meta['avg'],args.interval)
        end_geo=geometry_report(*last,str(calib))
        end_ok=geometry_passed(end_geo)
        log.event('静止采集完成，保持停在目标站',status='completed',
                  final_geometry=end_geo,final_geometry_passed=end_ok,
                  first_to_last_image_shift=[shift_px(a,b) for a,b in zip(first,last)],
                  end_encoder_deg=rig.motor.read_position_deg(),motor_status=rig.check_status(),
                  last_known_position_pulses=rig.pos,position_uncertain=False)
    except BaseException as exc:
        if rig:
            rig.stop()
        log.event(f'固定站采集中止，不自动回程：{exc}',status='aborted',error=repr(exc),
                  position_uncertain=bool(rig and rig.faulted))
        raise
    finally:
        if rig:
            rig.close(home=False)
    return out


if __name__=='__main__':
    main()
