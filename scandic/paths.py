# -*- coding: utf-8 -*-
"""项目路径与源码溯源的唯一来源。

重构后各模块位于 scandic/<子包>/, 不能再用 dirname(__file__) 推项目根。
统一从这里取: 采集数据 CAPTURES、WORK_LOG、以及源码哈希(按文件名在包树里
解析, 因为 dic_engine 与各处理模块不在同一子目录)。
"""
import hashlib
from pathlib import Path

PKG = Path(__file__).resolve().parent            # .../scandic
PROJECT_ROOT = PKG.parent                         # 项目根
CAPTURES = PROJECT_ROOT / "captures"
WORK_LOG = PROJECT_ROOT / "WORK_LOG.md"


def pkg_file(name):
    """把裸源文件名解析成包树内的真实路径(供溯源哈希)。"""
    hits = list(PKG.rglob(name))
    if not hits:
        raise FileNotFoundError(f"包内找不到源文件 {name}")
    return hits[0]


def hash_named(names):
    """{文件名: sha256} —— 指定若干源文件(校验匹配算法是否变更)。"""
    return {n: hashlib.sha256(pkg_file(n).read_bytes()).hexdigest() for n in names}


def hash_all_sources():
    """{文件名: sha256} —— 整个 scandic 包的全部 .py(写入 manifest 溯源)。"""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PKG.rglob("*.py")) if "mvsdk" not in p.parts}
