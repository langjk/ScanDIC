# ScanDIC — 双目扫描式三维 DIC

当前主线是双目工业相机 + ZDT 电动滑台：基准扫描 → 加载 → 按相同站位
复扫 → 逐站三维 DIC → 拼接整板位移场。扫描原点由板边纹理确定。
早期不可控滑台的单相机全景拼接功能仍保留，见后文旧设备部分。

开发、验证及实机扫描进度持续记录在 [WORK_LOG.md](WORK_LOG.md)。

2026-09-21：设备已下线，本轮只处理历史图像和离线模拟。扫描和固定站处理默认
采用三次样条亚像素匹配、左右相机分别做时序跟踪；不自动扣除整体位移。
整改证据、同点比较、适用边界和复验步骤见
[离线误差整改报告](captures/offline_error_remediation/report.md)。

## 项目结构 (2026-10-16 重构为包)

代码已整理为 `scandic/` 包，运行一律在**项目根**执行 `python -m <模块路径>`：

```
scandic/
  paths.py            项目路径与源码溯源的唯一来源
  dic_engine.py       实时三维 DIC 核心引擎
  hardware/           dahua_cam · zdt_motor · mvsdk/(相机 SDK)
  calibration/        stereo_calibrate · calib_capture
  capture/            scan_capture · capture_static_station · resume_reference_scan
  processing/         scan_pipeline · precision_scan_pipeline · precise_match ·
                      stereo_temporal · offline_dic · camera_motion · scan_report ·
                      stitch · strain_field · scan_process · process_static_station
  gui/                calib_gui · live_dic_gui · focus_assist
  diagnostics/        diagnose_epipolar · diagnose_camera_motion ·
                      diagnose_calib_live · reset_trigger
tests/                42 项回归
legacy/               旧单相机广角全景管线(ALI121 设备)
archive/              退役的一次性过程脚本(历史留档)
reference/            硬件手册 PDF / CAD / MVviewer 安装包(不入库)
captures/             采集数据与结果(不入库)
```

主要入口命令（旧 `python X.py` → 新 `-m` 形式）：

| 用途 | 命令 |
|---|---|
| 双目标定 GUI | `python -m scandic.gui.calib_gui` |
| 实时三维 DIC | `python -m scandic.gui.live_dic_gui` |
| 基准/变形扫描 | `python -m scandic.capture.scan_capture ref/def ...` |
| 扫描离线处理 | `python -m scandic.processing.scan_process <目录>` |
| 应变场(Green-Lagrange) | `python -m scandic.processing.strain_field <precision_results>` |
| 立体标定(命令行) | `python -m scandic.calibration.stereo_calibrate` |
| 固定站处理 | `python -m scandic.processing.process_static_station` |
| 回归测试 | `python -m unittest tests.test_precision tests.test_scan_validation tests.test_scan_pipeline tests.test_static_station` |
| 旧设备全景 | `python -m legacy.scan_gui` / `python -m legacy.headless_scan` |

下文命令块已按此更新；正文中 `foo.py` 指模块名，实际运行路径见上表。

## 环境

本机核验：Python 3.10.11、`opencv-python-headless` 4.11.0.86、`numpy` 2.1.2、
`scipy` 1.15.3、`PyQt5` 5.15.11、`pyserial` 3.5。相机另依赖 MV Viewer 的 x64 SDK。

## 新设备: 大华 MV-A7A20MU30 双目 (2026-09-11 起)

两台华睿 USB3 工业相机(4096×3000 Mono8 @30fps, 全局曝光), 序列号
BK24404AAK00007 / BK06664AAK00010。SDK 为 MV Viewer 自带 Python 接口:

- `mvsdk/MVSDK.py` — 从安装目录拷贝的官方 ctypes 封装, 打了两处补丁:
  ① 改为加载 64 位 `Runtime\x64\MVSDKmd.dll`(原文件写死 32 位相对路径);
  ② **加载 DLL 时必须临时把 cwd 切到 DLL 目录**, 否则其 DllMain 按相对
  路径找配置会报 WinError 1114。
- `dahua_cam.py` — 项目封装: 枚举/按序列号打开/锁曝光/取流出 numpy 灰度图。
  注意 MV Viewer 软件占用相机时 connect 会失败, 用前先关闭或断开。
- `focus_assist.py` — 调焦辅助(终端版), 拧对焦环看实时清晰度数值。
- `calib_gui.py` — **双目标定 GUI**(首选): 双路实时预览、清晰度/亮度/饱和
  指标(调光圈对焦)、自动曝光开关、圆点/棋盘实时检测状态灯、一键采集
  标定对(双目均检出才存)、自动采集模式、界面内直接跑 stereo_calibrate。
- `calib_capture.py` / `stereo_calibrate.py` — 命令行版采集与立体标定;
  支持圆点阵(默认)与棋盘格, 已处理对称圆点阵 180° 旋转二义性。
  圆点检测为两级(1/4 图定网格 + 全分辨率灰度质心精定位, 全分辨率直接
  findCirclesGrid 会大面积失败; 质心法对空心环形基准点同样成立)。
  内置两道自动质量闸门: 单视图重投影峰值>1.5px(质心被遮挡/反光带偏)
  和左右相对位姿偏离中位(触发滑帧)的对自动剔除。
- **标定 GUI 必须用软触发同步对拍**(TriggerPump): 自由流各取"最新帧"
  左右可差一个帧周期, 手持标定板晃动会把立体 RMS 打到像素级。
- **最新已保存标定**: `captures/stereo_calib_20260920_154733/stereo_calib.npz`
  (2026-09-20, 13 对, 立体 RMS 0.326px, 基线 228.694mm,
  夹角 24.453°, 左目 fx≈15681px)。扫描前须用当前散斑图核验极线残差。
  9 月 18 日标定已在 9 月 20 日旧预检中发现与机架不符（极线中位约 58px）。
  9 月 14 日的 0.228px / 322.628mm / 32.925° 是旧构型结果。
  相机对焦/光圈/相对位姿一动就要重新评估并重标。
- `reset_trigger.py` — 软触发残留导致 MV Viewer 取流 0 帧时跑一次恢复。
- 实测两台均可 4096×3000 @ 30fps 稳定取流; 软触发双机对拍 15 对/秒,
  触发时差 ~0.3ms。

## 实时三维 DIC (live_dic_gui.py / dic_engine.py)

```
python -m scandic.gui.live_dic_gui    # 打开相机 -> 设为参考 -> 变形, 云图实时刷新
```

- **不做平面立体校正(unrectified)路线**: 旧构型长焦(FOV~15°)+汇聚角 33°
  曾出现 stereoRectify 映射全黑；这属于当时构型/参数下的实测问题，
  不代表立体校正在一般几何上不可行。目前继续采用: 去畸变 -> 测量面单应(SIFT+RANSAC 自举)把右图
  预扭到左视角 -> DIS 稠密光流匹配小残差 -> 逐像素线性三角化。
- 位移输出在试件平面坐标系(U/V 面内, W 离面), 单位 mm。
- 历史实测(旧构型静止场景): 极线误差 0.23px(与当时标定 RMS 一致), 噪声底
  U 1.5 / V 0.8 / W 3.3 µm(中位), ~3.4fps。散斑稀疏区噪声偏大。
- 精度定位: 光流级(~0.1-0.3px), 供实时云图与现场判断;
  定量分析需用原始双目图像离线做子区 DIC。"保存当前场"仅保存位移和掩码，
  不含可重新匹配的原始双目图；应使用序列采集保存图像。
- 抗反光 = 局部对比度归一化(保亚像素梯度, 未用二值化) + 大块饱和
  连通域剔除(单个饱和散斑白点正常, 不剔)。
- **长焦立体对位姿漂移极敏感**: fx≈15500 下相机转 0.006° 即 1.6px
  极线误差, 有效区骤减(实测踩过: 换试件碰了机架 → "没场")。
  "设为参考"时自动做标定有效性自检(SIFT 内点对标定 F 的残差),
  >0.8px 会警告要求重标。测量期间勿碰相机杆/云台; 画面过暗(照明
  故障)也会自动告警。

## 连续静态采集 + 离线高精度 DIC

- live_dic_gui 内"连续静态采集": 按间隔存多帧平均图像对(16bit PNG,
  保留亚灰阶)到 captures/seq_xxx, 每状态约 25MB。
- `offline_dic.py captures/seq_xxx`: 保留的旧序列处理器，仍使用整像素 ZNCC +
  亚像素相位相关，输出 results/(逐状态 npz + summary.csv + W 云图)。
  本轮精度整改接入 `scan_process.py` 和 `process_static_station.py`，旧序列入口尚未迁移，
  不应将其结果与新的三次样条处理结果混称。
- 2026-09-20 已修复子区残差符号和16位图读取/预处理的浮点精度，
  并保留 `--output` 另存结果。相位相关质心估计仍会受纹理频谱影响；
  本次平滑合成纹理的分量误差 P95 为0.2425px，不能宣称普适0.01px精度。
- 历史静态测试 `captures/seq_static_test2/results/summary.csv`（修复前版本）：
  4 帧平均、64px 子区/32px 网格；U/V/W 的 P95 绝对值分别为
  6.22–7.97 / 3.27–3.43 / 32.48–43.55um。静态重复性不等同于测量准确度；
  已发现旧匹配器亚像素残差符号错误，修复后需另存结果重新验证。
  W 对几何敏感(dZ/dd=0.083mm/px), 想再压: 加大 --subset / 增大基线夹角。
- GUI 启动时选择最新标定；离线处理读取采集目录 `meta.json` 绑定的标定，
  避免用新机架标定覆盖旧数据的几何关系。

## 滑台电机 (ZDT X42S, Emm 固件, COM4)

- `zdt_motor.py`: 串口驱动(115200, 地址1, 校验0x6B)。enable /
  move_pulses(±脉冲, 正CW负CCW, 3200脉冲=1圈) / stop / read_status
  (含到位、堵转、左右限位) / read_position_deg / wait_arrival。
- 实测: +3200脉冲=+360.8°(Emm固件确认), ±20000往返3s@400RPM, 闭环归位良好。
- 2026-09-20 用户更新：取消累计脉冲预算，滑台机械边缘以堵转保护停止。
  软件运动中持续检查堵转/限位，长距离分段下发，异常后不自动回程。
- 运行前确认张大头上位机已关闭串口(独占 COM4)。

## 扫描式 DIC (scan_capture.py / scan_process.py)

范式: 基准扫描 -> 加载变形 -> 复扫相同站位 -> 逐站立体 DIC -> 拼整板。

```
python -m scandic.capture.scan_capture ref --seek-edge --auto-edge --overlap 0.6
python -m scandic.capture.scan_capture def --plan captures/scan_ref_xxx
python -m scandic.processing.scan_process captures/scan_def_xxx
```

- 最终编排：先在当前位置小位移探测 px/脉冲及方向；反向按 30% 视场
  步进，纹理覆盖率首次低于 80% 时回退一步，将板内最后一站设为原点。
  从原点正向扫描，站距按重叠率计算；另一侧纹理覆盖率低于 `--edge-frac`
  （默认 55%）时收尾。正常结束回原点，在板边待命；不寻滑台机械端。
- 寻边覆盖率使用四分之一图上的相对纹理 `std/(mean+3)>0.10` 且 `std>1.5`，
  避免固定亮度阈值把暗散斑误认为板外。它仍是图像判据，需核验保存的两端诊断图。
- 相邻图纯平移NCC不足时，以SIFT单应补偿实机跨站透视/缩放后核验重叠NCC；
  位移在图像中心求值，只用于采集粗定位，不替代三维DIC。
- 变形复扫用图像配准迭代对准基准站位0，默认误差须低于 **0.3px**，
  使用4帧平均，最多4次修正；`--alignment-tolerance-px` 可显式配置。
  NCC过低、迭代未收敛或存在单轴滑台不能修正的横向偏差时中止。
  `--alignment-roi X Y W H` 指定原始左图内已知静止参照，宽高均至少128px。
  加载试验应使用独立于变形试件的固定参照；若以变形区域对齐，会混入试件真实运动。
  对齐容差不是测量准确度承诺，这些改动目前仅通过离线模拟。
  只能在与基准图仍有足够重叠时恢复原点；电机记忆不是对齐依据。
- 默认不限制累计脉冲；`--budget` 仅作为可选显式限制保留，无40000硬上限。
  长运动按不超过8000脉冲分段，运动中查堵转/限位并验收编码器。采用手册p57的
  mode=2（相对实时位置）避免沿用中断前目标。异常停止、不自动回程。
- `--auto-edge` 默认无5站截断；堵转/限位、可选预算或显式站数上限先触发时标为中止，
  不宣称已扫到板边。正常结束的回程检查仍为图像残差<30px，
  它只作行程收尾核验，不是下次复扫对齐或计量合格判据。
- 每个阶段/站位同步 `WORK_LOG.md`、采集目录 `scan_log.md` 和原子更新的
  `meta.json`；标定复制绑定到采集目录，每站检查极线误差（全分辨率中位≤0.8px）。
  `station_-1_*` / `station_-2_*` 为两端板边诊断图，不属于复扫站位表。
  质量检查前先存 `candidate_XX_*`，通过后转为 `station_XX_*`，失败保留候选图。
- SIFT极线中位超过0.8px时，改为先做固定网格子区/ECC复核，不直接判重标。
  复核匹配不使用F约束；至少100个可靠子区，中位≤0.8px且P95≤1.5px才通过。
  未改变中位门槛，也未从散斑平面重新拟合标定来消除误差。
- 几何闸门中止且位置已知时，可用 `python -m scandic.capture.resume_reference_scan <ref或def目录>` 续扫；
  启动前核对编码器、曝光和双目图像，备份候选图与元数据，不覆盖已接受站位。
- 回归验证：`python -X utf8 -B -m unittest tests.test_precision tests.test_scan_validation tests.test_scan_pipeline tests.test_static_station -v`，
  42项通过，包含非零位移保持、锚点补偿不吞掉真实变形、起点ROI与模拟硬件异常处理。
- 处理端：`scan_process.py` 调用 `precision_scan_pipeline.py`。
  `precise_match.py` 做三次样条插值迭代匹配，联合消除局部亮度增益/偏置影响。
  `stereo_temporal.py` 只在参考时刻建立一次立体对应，然后分别跟踪左右相机的同一纹理点。
  跨站先单应预扭补偿透视，再用精确匹配估计局部残差，
  对下一站实际对应像素三角化；RANSAC/Kabsch建立刚体链，不以最近网格点代替。
  缺站、标定/计划不一致或不可靠连接均显式失败，不复用前站变换。
- 默认64px子区、64px采样步长、16px搜索半径。输出 `precision_scan_results/` 下的
  `merged.npz`、`station_XX.npz`、站间对应、JSON统计、`report.md`、运行状态和源码哈希。
  输出目录已存在时拒绝覆盖，用 `--output 新目录` 另存。
  **默认不做任何整体归零或重叠平移修正**：`D == D_raw`，`uvw == uvw_raw`。
  无独立静止参照时，这是相机系下的表观位移，相机运动与试件刚体运动不可分离，
  `absolute_displacement_identifiable=false`。
  `--overlap-diagnostic` 另存 `uvw_overlap_diagnostic`，不替代正式位移。
  统计按有效采样点加权（保留重复覆盖），不是均匀面积权重；接缝误差也不是独立准确度。
- 验证方法: 板子不动扫两轮, 整板位移场应处处~0。
- 最新实测（2026-09-20）：`captures/scan_ref_20260920_184347` 完成14站基准采集，
  两侧板边已确认，已返回原点（图像残差约4.01px、NCC0.999）。站13曾因SIFT统计
  超线中止，经独立子区/ECC核验后续扫；详情见该目录 `scan_summary.md`。
  不加载复扫 `scan_def_20260920_194648` 已采齐14站，28张16位图解码核验通过；
  已回原点并完成零位移分析，29623点，覆盖约831.5×99.0mm条带。
  原始U/V/W绝对P95=250.54/7.97/31.74μm；重叠修正后=148.50/5.69/46.73μm。
  U存在显著整体偏差，W经平移链后累计漂移增大；本次未预设合格阈值，
  不能宣称准确度验证通过。详细统计与下一步见
  [零位移验证报告](captures/scan_def_20260920_194648/scan_results/zero_validation.md)。

### 离线整改后的实测基线

同一像素位置、同一旧试件坐标系下，绝对P95（U / V / W，μm）：

|数据|旧方法|新方法|样本数|
|---|---|---|---:|
|固定站11，20组相对状态0|3.718 / 3.216 / 11.113|1.178 / 2.306 / 6.246|41262|
|14站不加载复扫|250.381 / 7.935 / 31.424|252.554 / 11.695 / 12.245|29094|

扫描U整体偏差仍在，V未改善；不能把算法更精确等同于所有原始零偏都消失。
固定站采集的到站/结束几何核验未通过（ECC约1.204/1.189px），所以表内三维值仍是诊断换算。
在明确试件静止的零位移数据上，以384px交替空间块拟合相机位姿，未参与拟合的14879点
U/V/W绝对P95为0.568/0.587/2.668μm；这是有条件的诊断，不能移用为加载试验的逐站归零。

### 独立静止锚点与结果含义

`--anchors anchors.json` 才会启用相机运动补偿。配置必须显式声明
`known_stationary: true`，并为**每站**给出 `stationary_regions`：键是站号字符串，
值为矩形列表 `[[x0,y0,x1,y1], ...]`（右下界不包含）。坐标属于
**去畸变后的参考左图**，与采集对齐ROI的原始图坐标不同。
每站至少30个有效、具有二维分布的锚点；锚点必须固定于外部稳定参照，不能随试件变形。
软件能验证数量、分布和拟合收敛，不能从图像自己证明锚点确实静止。
补偿后仍保留 `D_raw`，`D` 才是锚点条件下的结果。需另做已知非零位移实验验证准确度。

```powershell
# 保存的14站数据：新目录重算，完全离线
python -X utf8 -B -u -m scandic.processing.scan_process captures/scan_def_20260920_194648 --output precision_scan_rerun

# 使用本次已核验缓存，可省去重复的站内匹配；输出目录必须尚不存在
python -X utf8 -B -u -m scandic.processing.scan_process captures/scan_def_20260920_194648 --cache captures/scan_def_20260920_194648/precision_results --output precision_scan_cached_rerun

# 固定站静止分析：该批几何失败，必须显式标为诊断
python -X utf8 -B -u -m scandic.processing.process_static_station captures/static_station11_20260920_203311 --diagnostic --output precision_static_rerun
```

固定站处理默认 `--matcher precise`；`--matcher legacy` 仅用于对照。
可用 `--cache captures/static_station11_20260920_203311/precision_results` 复用本次静止匹配缓存。
`precision_results/` 是本次A/B对照缓存，`precision_scan_results/` 是新扫描拼接输出，
`precision_static_results/` 是新静止时序报告；历史 `scan_results/` 和 `results/` 均保留。

## 旧设备: 相机(ALI121-Y 车载 GMSL 模组)与去畸变

旧相机: f=1.75mm / HFOV 129.6° 广角、原生 1280×960(GMSL 桥拉伸输出 1920×1080)、
卷帘快门、序号 1(DSHOW)。**焦点没问题**(21cm 工作距实测中心清晰度充足,
散斑 4px+), 但广角畸变会破坏纯平移拼接模型——2026-09-10 实拍验证:
不去畸变时位移一大 NCC 即崩(0.88→0.10), 去畸变后 266px 位移 NCC=0.91。

- 去畸变参数在 `undistort.json`(等距鱼眼模型, 由实拍位移场拟合, 非棋盘格标定),
  `calib.py` 负责加载; `scan_gui.py` / `headless_scan.py` 自动套用, 并只取
  画面**中央 60%** 参与拼接(边角软、残余畸变大)。
- **相机位置/距离变了不需要重标**(畸变属于镜头), 但换相机必须重新拟合:
  录一段慢速平移(`record_raw.py`), 用两张相距 200~300px 的清晰帧跑
  scratchpad 的 fit_v2.py 流程重新拟合。
- **推滑台必须慢**: 曝光 -6 档≈15.6ms, 图像速度 v px/s 产生 v×0.0156 px 运动模糊;
  实测 4000px/s 时模糊 60px 直接配准丢失。全程匀速 15~30 秒推完最稳。

## 旧设备流程: 实时全景扫描(scan_gui.py)

```
python -m legacy.scan_gui
```

1. 相机序号选 **1**（外接 UVC 相机, 1920×1080 @ ~18fps; 0 是另一个 720p 设备），点“打开相机”。
2. **调焦**: 看状态栏“清晰度”数值, 调节镜头使其尽量大（虚焦时只有个位数, 对好焦通常几百以上）。
3. **锁曝光**: 取消勾选“自动曝光”, 用滑块调到不过曝、不欠曝。曝光时间尽量短, 避免运动模糊。
4. 点“**开始扫描**”, 然后让滑台滑动（先点按钮后动滑台, 顺序无所谓, 静止帧会被自动跳过）。
   - 状态灯: 绿=跟踪/落关键帧, 黄=滑台静止, **红=配准丢失**（滑台过快、失焦或无纹理, 放慢即可自动恢复）。
   - 下方全景条实时更新（1/4 分辨率预览）。
5. 扫完点“**结束扫描**”, 后台做全分辨率羽化融合（每关键帧几十毫秒量级）, 保存到
   `captures/scan_时间戳/`:
   - `panorama.png` — 整板全景（融合时自动做平场校正, 照明分布由整个扫描的移动帧均值估计）
   - `positions.csv` — 每个关键帧在全景中的亚像素位置（后续 DIC 把局部帧映射回整板坐标直接用）
   - `key_00000.png ...` — 各关键帧原图（可用 stitch.py 离线重拼/复核）

**滑台速度上限**: 处理线程始终只取最新帧（跟踪约 15–25 帧/秒）, 两次处理之间的位移
需小于半个视场。1080p 下即滑台在图像上的速度 < ~7000 px/s——一般滑台远达不到, 不用担心;
真正常见的问题是**运动模糊**, 靠缩短曝光解决。

## 工作原理

- **跟踪 + 关键帧**: 每帧在 0.5× 降采样高通图上与上一关键帧做相位相关(快速);
  位移超过 35% 视场时全分辨率亚像素精配, 落为关键帧。
- 扫描中只做轻量工作: 关键帧由后台线程存盘, 预览拼在 1/4 小画布;
  全分辨率融合在结束时一次完成。
- 配准一律在**高通滤波图**上做: 暗角/照明梯度会给相位相关带来 ~0.2px/对 的系统偏置。
- 每对配准用重叠区 NCC 校验, 不合格的帧不会进入全景。

## 无界面自动扫描(headless_scan.py) — 已实拍验证

```
python -m legacy.headless_scan            # 启动后推滑台, 推完停 8 秒自动结束
```

自动: 锁曝光、去畸变、中央裁剪、等待运动、位移不再变化 8 秒后收尾、
结束时把不足关键帧阈值的尾段补成最后一个关键帧。输出同 scan_gui。
2026-09-10 实拍: 6 关键帧 / 全景 3170×655 / 零丢失 / 关键帧间 NCC 0.78-0.79。

`record_raw.py`: 纯录制(帧差检测运动, 存全部原始帧), 诊断/重标定用。

## 备用: 离线采集+拼接

先录后拼（`capture_gui.py` 录制 → `stitch.py` 拼接), 或对实时扫描存下的关键帧重拼:

```
python -m scandic.processing.stitch captures/scan_xxxx -o pano.png --flat   # 重拼关键帧
python -m legacy.capture_gui                                                # 纯录制
python -m scandic.processing.stitch captures/run_xxxx -o panorama.png --flat # 离线拼接
```

`stitch.py` 参数: `--flat` 平场校正(推荐) / `--gray` 灰度输出 / `--min-shift` 静止阈值 /
`--min-ncc` 质量下限 / `--blend-power` 融合权重次幂(默认 4)。

**融合务必用"中心占优"权重(blend_weight, 高次幂帐篷函数)**: 宽重叠区若做
接近 50/50 的平均, 帧间 1~2px 残余错位会把散斑边缘抹虚——实拍中全景锐度
掉到关键帧的 0.51, 改中心占优后恢复到 0.88。亚像素平移用 INTER_CUBIC。

## 验证结果(合成散斑数据)

- 离线: 11 帧扫 ~3800px, 累计误差 max 0.55px / RMS 0.26px; 37% 低重叠 max 0.50px;
  全景 vs 原板整幅 NCC 0.985。
- 实时管线: 模拟不可控滑台(加速/抖动匀速/中途停顿)160 帧扫 ~6100px,
  9 关键帧累计漂移 max 1.3px(0.02% 比例误差), 分段局部 NCC 0.986–0.997, 无可见接缝;
  跟踪中位 ~50–70ms(相机 18fps=56ms/帧), 关键帧 ~150–300ms(期间自动丢帧, 不影响跟踪)。

## 踩过的坑(改代码前必读)

- `phaseCorrelate` 的历史实现陷阱（原记录标注 OpenCV 5.0，本机当前是 4.11），
  `stitch.py` 保留以下防护:
  1. 图像尺寸恰为 DFT 最优尺寸时会**原地加窗修改输入**——传入前必须 `.copy()`;
  2. **奇数DFT尺寸时峰值定位可偏0.5px**——只裁成偶数仍不充分，
     如122被内部补到125；现统一原区域加窗后补零到偶数最优DFT尺寸。
- 不为速度裁掉测量数据；`phase_correlate_even` 仅为避免奇数DFT偏置补零。
- 散斑图像高熵, PNG 高压缩级别极慢且省不了空间, 实时存盘用 `IMWRITE_PNG_COMPRESSION=1`。
- 本机是 `opencv-python-headless`, 没有 `cv2.imshow`, 预览一律走 PyQt5;
  相机用 DSHOW 后端(MSMF 在本机不可按序号枚举)。
