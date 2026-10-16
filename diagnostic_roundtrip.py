"""在已有相邻站位之间做诊断往返；启动前核对编码器和双目图像。"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from scan_capture import Rig, ScanLog, motion, save_pair, shift_px, require_shift, geometry_report


def load_reference(directory, index, tag):
    paths = [directory/f'{prefix}_{index:02d}_{tag}.png' for prefix in ('station', 'candidate')]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise ValueError(f'Missing reference for {index} {tag}')
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(str(path))
    return a.astype(np.float32)/(256 if a.dtype == np.uint16 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('--from-station', type=int, required=True)
    args = ap.parse_args()
    d = Path(args.directory).resolve()
    meta = json.loads((d/'meta.json').read_text(encoding='utf-8'))
    k = args.from_station
    if k < 1 or meta.get('position_uncertain') or meta['last_known_position_pulses'] != meta['origin_pulses']+k*meta['step_pulses']:
        raise ValueError('Reference stop position does not match requested station')
    refs = {i: [load_reference(d, i, tag) for tag in ('L', 'R')] for i in (k-1, k)}
    expected_deg = meta['start_encoder_deg'] + meta['last_known_position_pulses']*360/3200
    out = d/time.strftime('roundtrip_%Y%m%d_%H%M%S')
    log = ScanLog(str(out), {'role': 'diagnostic_roundtrip', 'status': 'initializing',
                            'ref_dir': str(d), 'logical_stations': [k, k-1, k],
                            'records': [], 'returned_to_start': False})
    rig = None
    try:
        rig = Rig(budget=None)
        if abs(rig.start_deg-expected_deg) > 3.6:
            raise RuntimeError('Encoder does not match last known stopped position; no motion allowed')
        log.event('诊断往返：已核对编码器，先拍当前位置再决定是否运动', status='running')
        for i, station in enumerate([k, k-1, k]):
            if i:
                target = -meta['step_pulses'] if station == k-1 else 0
                motion(rig, log, target, f'诊断到站 {station}')
            a, b = rig.pair_avg(meta['avg'])
            save_pair(str(out), i, a, b)
            shifts = [shift_px(old, cur) for old, cur in zip(refs[station], (a, b))]
            for dx, dy, ncc in shifts:
                require_shift(dx, dy, ncc, minimum=0.8)
                if np.hypot(dx, dy) >= (2 if i == 0 else 30):
                    raise RuntimeError('Image position differs from saved station')
            geo = geometry_report(a, b, meta['calib'])
            record = {'image_index': i, 'logical_station': station, 'encoder_deg': rig.motor.read_position_deg(),
                      'image_shift_from_saved': shifts, 'geometry': geo}
            log.meta['records'].append(record)
            log.event(f'诊断站 {station}: {json.dumps(record, ensure_ascii=False)}')
        log.event('诊断往返完成，回到站位13；未继续正向扫描',
                  status='completed', returned_to_start=True)
    except BaseException as exc:
        if rig:
            rig.stop()
        log.event(f'诊断中止，不自动回程: {exc}', status='aborted', error=repr(exc),
                  position_uncertain=bool(rig and rig.faulted))
        raise
    finally:
        if rig:
            rig.close(home=False)


if __name__ == '__main__':
    main()
