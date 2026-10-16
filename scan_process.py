"""扫描DIC入口：精确双时序匹配；无静止锚点时保留表观整体位移。"""
from precision_scan_pipeline import main
from scan_pipeline import kabsch, load_station

if __name__ == '__main__':
    main()
