# -*- coding: utf-8 -*-
"""
ZDT X42S(Emm42_V5 协议) 闭环步进电机串口驱动 —— 滑台轴控制。

协议依据: ZDT_X42S第二代闭环步进电机用户手册V1.0.5 (项目根目录 PDF)。
串口 TTL 115200, 地址 1, 校验字节固定 0x6B。

用法:
    from scandic.hardware.zdt_motor import ZdtMotor
    m = ZdtMotor("COM4")
    m.enable()
    m.move_pulses(+20000, rpm=300, acc=100)   # 正=CW, 负=CCW (细分16: 3200脉冲/圈)
    m.wait_arrival()
    print(m.read_status(), m.read_position_deg())
"""
import time

import serial

MAX_MOVE_PULSES = 40000


class ZdtMotor:
    def __init__(self, port="COM4", baud=115200, addr=1, timeout=1.0):
        self.addr = addr
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self):
        self.ser.close()

    # ---------- 底层 ----------

    def _xfer(self, payload, resp_len):
        """发送 addr+payload+6B, 读定长应答(自动跳过缓冲区中的杂散帧)。"""
        frame = bytes([self.addr]) + bytes(payload) + b"\x6b"
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        want_func = payload[0]
        buf = b""
        t0 = time.time()
        while time.time() - t0 < self.ser.timeout + 0.5:
            b = self.ser.read(1)
            if not b:
                continue
            buf += b
            # 对齐帧头 addr+func
            i = buf.find(bytes([self.addr, want_func]))
            if i >= 0 and len(buf) - i >= resp_len:
                frame = buf[i:i + resp_len]
                if frame[-1] != 0x6B:
                    raise RuntimeError("电机应答结束校验字节不是 0x6B")
                return frame
            if len(buf) > 64:
                buf = buf[-8:]
        raise TimeoutError(f"电机无应答(func=0x{want_func:02X})")

    # ---------- 命令 ----------

    def enable(self, on=True):
        r = self._xfer([0xF3, 0xAB, 1 if on else 0, 0x00], 4)
        if r[2] != 0x02:
            raise RuntimeError(f"使能失败, 返回 0x{r[2]:02X}")

    def move_pulses(self, pulses, rpm=300, acc=100, mode=0):
        """位置模式(Emm): pulses 正=CW 负=CCW。

        mode 0=相对上次目标，1=绝对位置，2=相对当前实时位置（手册 p57）。

        细分16 时 3200 脉冲 = 1 圈。acc 0=直接启动, 1-255 档位越大越快。
        """
        d = 0 if pulses >= 0 else 1
        p = abs(int(pulses))
        if p > MAX_MOVE_PULSES:
            raise ValueError(f"单次运动超过 {MAX_MOVE_PULSES} 脉冲硬上限")
        rpm = int(rpm)
        if mode not in (0, 1, 2) or not 1 <= rpm <= 3000 or not 0 <= acc <= 255:
            raise ValueError("无效的运动模式、速度或加速度")
        pay = ([0xFD, d, (rpm >> 8) & 0xFF, rpm & 0xFF, acc & 0xFF]
               + [(p >> s) & 0xFF for s in (24, 16, 8, 0)]
               + [mode & 0xFF, 0x00])
        r = self._xfer(pay, 4)
        if r[2] == 0xE2:
            raise RuntimeError("电机拒绝运动(未使能/堵转保护中?)")
        if r[2] not in (0x02, 0x9F):
            raise RuntimeError(f"位置命令返回 0x{r[2]:02X}")

    def stop(self):
        r = self._xfer([0xFE, 0x98, 0x00], 4)
        if r[2] != 0x02:
            raise RuntimeError(f"停止失败, 返回 0x{r[2]:02X}")

    def read_status(self):
        r = self._xfer([0x3A], 4)
        f = r[2]
        return {"enabled": bool(f & 0x01), "in_position": bool(f & 0x02),
                "stalled": bool(f & 0x04), "stall_protect": bool(f & 0x08),
                "limit_left": bool(f & 0x10), "limit_right": bool(f & 0x20)}

    def read_position_deg(self):
        """实时位置(度)。本机为 Emm 固件(实测+3200脉冲=+360.8°): 65536=一圈。"""
        r = self._xfer([0x36], 8)
        sign = -1 if r[2] == 1 else 1
        raw = int.from_bytes(r[3:7], "big")
        return sign * raw * 360.0 / 65536.0

    def wait_arrival(self, timeout=60.0, poll=0.05, target_deg=None,
                     tolerance_deg=3.6):
        """持续查故障；可同时核验编码器，避免命令后旧到位标志提前返回。"""
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < timeout:
                st = self.read_status()
                if any(st[k] for k in ("stalled", "stall_protect", "limit_left", "limit_right")):
                    raise RuntimeError(f"堵转或限位触发: {st}")
                if not st["enabled"]:
                    raise RuntimeError("运动过程中电机失能")
                if st["in_position"]:
                    if target_deg is None or abs(self.read_position_deg() - target_deg) <= tolerance_deg:
                        return True
                time.sleep(poll)
            raise TimeoutError("等待到位超时")
        except BaseException:
            try:
                self.stop()
            except Exception:
                pass
            raise


if __name__ == "__main__":
    m = ZdtMotor("COM4")
    print("状态:", m.read_status())
    print("位置(Emm解, X解):", m.read_position_deg())
    m.close()
