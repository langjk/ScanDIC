"""Audit saved offline outputs and build the September 2026 error-remediation report."""
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import scipy

# 已退役的 9 月整改报告生成器(历史留档)。路径已指向项目根。
from scandic.paths import PROJECT_ROOT as ROOT, CAPTURES as CAP
REF=CAP/'scan_ref_20260920_184347'
SCAN=CAP/'scan_def_20260920_194648'
STATIC=CAP/'static_station11_20260920_203311'
OUT=CAP/'offline_error_remediation'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path) as z:
        return {key:z[key] for key in z.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p95(s):
    return [s[c]['p95_abs_um'] for c in 'UVW']


def numbers(s):
    return ' / '.join(f'{v:.3f}' for v in p95(s))


def check_statistics(values,expected):
    assert len(values)==expected['count']
    assert np.isfinite(values).all()
    for j,c in enumerate('UVW'):
        v=values[:,j]*1000
        actual=[np.median(v),v.mean(),np.sqrt(np.mean(v*v)),np.percentile(abs(v),95)]
        target=[expected[c][k] for k in ('median_um','mean_um','rms_um','p95_abs_um')]
        np.testing.assert_allclose(actual,target,rtol=1e-10,atol=1e-9)


def audit():
    historical={}
    for d,filename,key in ((SCAN,'data_verification.json','files'),
                            (STATIC,'results/data_verification.json','images')):
        historical.update({str(d/row['file']):row['sha256'] for row in read(d/filename)[key]})
    records=[]
    for d in (REF,SCAN,STATIC):
        for path in sorted(d.glob('station_[0-9][0-9]_[LR].png')):
            digest=sha(path)
            if str(path) in historical:
                assert digest==historical[str(path)],f'Historical image hash changed: {path}'
            im=cv2.imread(str(path),-1)
            assert im is not None and im.shape==(3000,4096) and im.dtype==np.uint16
            records.append(dict(file=str(path),sha256=digest,historical_sha256_checked=str(path) in historical))
    assert len(records)==98 and len(historical)==70
    oldmanifest=read(SCAN/'scan_results/manifest.json')
    for path,size,mtime in oldmanifest['input_images']:
        s=Path(path).stat()
        assert (s.st_size,s.st_mtime_ns)==(size,mtime)
    for d in (REF,SCAN,STATIC):
        assert sha(Path(read(d/'meta.json')['calib']))==oldmanifest['calibration_sha256']
    comparisons={}
    for d,mode,oldfolder in ((SCAN,'scan','scan_results'),(STATIC,'static','results')):
        summary=read(d/'precision_results/summary.json')
        oldsummary=read(d/oldfolder/'summary.json')
        manifest=read(d/'precision_results/manifest.json')
        for name in ('precise_match.py','stereo_temporal.py'):
            assert sha(ROOT/name)==manifest['code'][name]
        raw=[];old=[];same_new=[];same_old=[]
        for row in summary['states']:
            k=row['index'];z=arrays(d/'precision_results'/f'station_{k:02d}.npz')
            prefix='station' if mode=='scan' else 'state'
            prior=arrays(d/oldfolder/f'{prefix}_{k:02d}.npz')
            basis=(np.array(oldsummary['R_to_station0'][k]).T@np.array(oldsummary['basis'])
                   if mode=='scan' else np.array(oldsummary['basis_local']))
            newD=z['D']@basis;oldD=prior['D']@basis
            newxy=np.column_stack([z['px'],z['py']]).astype(int)
            oldxy=np.column_stack([prior['px'],prior['py']]).astype(int)
            common,ni,oi=np.intersect1d(newxy[:,0]+4096*newxy[:,1],
                                      oldxy[:,0]+4096*oldxy[:,1],return_indices=True)
            assert len(common)>=200
            check_statistics(newD,row['raw'])
            check_statistics(newD[ni],row['common_new']);check_statistics(oldD[oi],row['common_old'])
            raw.append(newD);old.append(oldD);same_new.append(newD[ni]);same_old.append(oldD[oi])
        for key,values in (('raw',raw),('legacy',old),('common_new',same_new),('common_old',same_old)):
            check_statistics(np.concatenate(values),summary[key])
        comparisons[mode]=dict(comparisons=len(raw),matched_pixel_samples=sum(map(len,same_new)))
    merged=arrays(SCAN/'precision_scan_results/merged.npz')
    summary=read(SCAN/'precision_scan_results/summary.json')
    np.testing.assert_array_equal(merged['D'],merged['D_raw'])
    np.testing.assert_array_equal(merged['uvw'],merged['uvw_raw'])
    assert summary['absolute_displacement_identifiable'] is False
    check_statistics(merged['uvw'],summary['output'])
    check_statistics(merged['uvw_overlap_diagnostic'],summary['overlap_diagnostic'])
    for k in range(14):
        z=arrays(SCAN/'precision_results'/f'station_{k:02d}.npz')
        R,t=merged['rotations'][k],merged['translations'][k]
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R),1,atol=1e-10)
        np.testing.assert_allclose(merged['P'][merged['station']==k],z['P']@R.T+t,atol=1e-10)
        np.testing.assert_allclose(merged['D'][merged['station']==k],z['D']@R.T,atol=1e-10)
    for k,row in enumerate(summary['links']):
        pair=arrays(SCAN/'precision_scan_results'/f'link_{k:02d}_{k+1:02d}.npz')
        residual=np.linalg.norm(pair['B']@np.array(row['R']).T+row['t']-pair['A'],axis=1)
        np.testing.assert_allclose(residual,pair['residual_mm'],atol=1e-10)
        np.testing.assert_allclose(np.percentile(residual[pair['inlier']],95),row['p95_mm'],atol=1e-10)
    diagnostic=read(SCAN/'precision_results/camera_motion_diagnostic/summary.json')
    heldout=[];rawheldout=[];rigidheldout=[]
    oldbasis=read(SCAN/'scan_results/summary.json')
    assert len(diagnostic['stations'])==14
    for row in diagnostic['stations']:
        k=row['station'];z=arrays(SCAN/'precision_results'/f'station_{k:02d}.npz')
        diag=arrays(SCAN/'precision_results/camera_motion_diagnostic'/f'station_{k:02d}.npz')
        train=diag['train_mask'];test=~train
        xy=z['ref_correspondence'][:,:2]
        independently_built=((xy[:,0]//384+xy[:,1]//384).astype(int)%2)==0
        np.testing.assert_array_equal(train,independently_built)
        assert not np.any(train&test) and train.sum()==row['train_count'] and test.sum()==row['test_count']
        basis=np.array(oldbasis['R_to_station0'][k]).T@np.array(oldbasis['basis'])
        heldout.append(diag['uvw_anchored'][test]);rawheldout.append((z['D']@basis)[test])
        rigidheldout.append(diag['uvw_rigid_rig'][test])
        check_statistics(heldout[-1],row['corrected_test'])
    for values,key in ((heldout,'corrected_test'),(rawheldout,'raw_test'),(rigidheldout,'rigid_rig_test')):
        check_statistics(np.concatenate(values),diagnostic[key])
    static=read(STATIC/'precision_static_results/summary.json')
    cube=arrays(STATIC/'precision_static_results/common_time_series.npz')['uvw']
    check_statistics(cube.reshape(-1,3),static['common_pooled'])
    assert static['geometry_passed'] is False
    sources=[p for d in (REF,SCAN,STATIC) for p in [d/'meta.json',Path(read(d/'meta.json')['calib'])]]
    results=[p for d in (SCAN/'precision_results',SCAN/'precision_scan_results',STATIC/'precision_results',
                         STATIC/'precision_static_results',SCAN/'scan_results',STATIC/'results')
             for p in d.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')]
    checks=dict(status='passed',images=records,historical_image_hash_matches=len(historical),
                original_scan_size_mtime_matches=56,calibration_sha256=oldmanifest['calibration_sha256'],
                comparisons=comparisons,raw_global_offset_preserved=True,
                rigid_links_recomputed=True,heldout_count=sum(map(len,heldout)),
                train_test_disjoint=True,statistics_recomputed=True,static_geometry_qualified=False,
                sources_and_results_sha256={str(p):sha(p) for p in sorted(set(sources+results))},
                code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
                environment=dict(python=platform.python_version(),numpy=np.__version__,
                                 scipy=scipy.__version__,opencv=cv2.__version__))
    (OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    return checks


def make_figures(scan,static,pose,old,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,s,title in zip(axes,(static,scan),('Fixed station: diagnostic geometry','14-station scan: raw apparent displacement')):
        x=np.arange(3)
        for offset,key,label,color in ((-.18,'common_old','Legacy','#999999'),(.18,'common_new','Precise','#176B99')):
            bars=ax.bar(x+offset,p95(s[key]),.36,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=list('UVW'),ylabel='Absolute P95 (um)',title=title,yscale='log')
        ax.set_ylim(.6,550);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same reference pixels and coordinate basis; no fitted offsets removed')
    fig.savefig(OUT/'same_pixel_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for j,c in enumerate('UVW'):
        for key,label,color in (('raw_test','Raw','#777777'),('rigid_rig_test','Rigid stereo rig: 6 DOF','#D98A22'),
                               ('corrected_test','Independent camera poses: 12 DOF','#176B99')):
            axes[j].plot([r['station'] for r in pose['stations']],
                         [r[key][c]['p95_abs_um'] for r in pose['stations']],'o-',label=label,color=color,ms=3)
        axes[j].set(title=c,xlabel='Station',ylabel='Held-out absolute P95 (um)',yscale='log')
        axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Known-stationary zero test only; fitting excludes the plotted spatial tiles')
    fig.savefig(OUT/'heldout_camera_models.png',dpi=170);plt.close(fig)
    m=arrays(SCAN/'precision_scan_results/merged.npz')
    oldoff=np.array(old['correction_camera0_mm'])@np.array(old['basis'])*1000
    newoff=m['overlap_diagnostic_offsets']@m['basis']*1000
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(np.arange(1,14),[r['p95_mm']*1000 for r in old['links']],'o-',label='Legacy')
    axes[0].plot(np.arange(1,14),[r['p95_mm']*1000 for r in new['links']],'o-',label='Precise')
    axes[0].set(xlabel='Link ending at station',ylabel='3D residual P95 (um)',title='Reference geometry connection residual')
    axes[1].plot(oldoff[:,2],'o-',label='Legacy overlap chain')
    axes[1].plot(newoff[:,2],'o-',label='Precise overlap chain (diagnostic only)')
    axes[1].axhline(0,color='gray',lw=1)
    axes[1].set(xlabel='Station',ylabel='Added W offset (um)',title='Accumulating offsets are not absolute correction')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'stitching_diagnostics.png',dpi=170);plt.close(fig)
    benchmark=read(OUT/'matcher_benchmark.json')
    names=list(dict.fromkeys(r['texture'] for r in benchmark))
    fig,axes=plt.subplots(1,4,figsize=(14,3.8),layout='constrained')
    for ax,name in zip(axes,names):
        rows=[r for r in benchmark if r['texture']==name]
        for method,label in (('phase','Phase centroid'),('spline','Cubic iterative')):
            ax.plot([r['p95_component_error_px'][method] for r in rows],'o-',label=label)
        ax.set(title=name.replace('gaussian_sigma_','Gaussian sigma=').replace('real_station11','Real speckle crop'),
               xlabel='Known shift case',ylabel='Component error P95 (px)',yscale='log')
        ax.set_xticks(range(6),[str(i) for i in range(6)]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Synthetic shifts, including gain 1.05 + offset 3; nine query points per case')
    fig.savefig(OUT/'matcher_benchmark.png',dpi=170);plt.close(fig)


def make_report(checks):
    scan=read(SCAN/'precision_results/summary.json');static=read(STATIC/'precision_results/summary.json')
    pose=read(SCAN/'precision_results/camera_motion_diagnostic/summary.json')
    old=read(SCAN/'scan_results/summary.json');new=read(SCAN/'precision_scan_results/summary.json')
    sm=read(STATIC/'meta.json');series=read(STATIC/'precision_static_results/summary.json')
    make_figures(scan,static,pose,old,new)
    benchmark=read(OUT/'matcher_benchmark.json')
    real=[r for r in benchmark if r['texture']=='real_station11' and r['shift']!=[0,0]]
    oldrange=[min(r['p95_component_error_px']['phase'] for r in real),max(r['p95_component_error_px']['phase'] for r in real)]
    newrange=[min(r['p95_component_error_px']['spline'] for r in real),max(r['p95_component_error_px']['spline'] for r in real)]
    oldlinks=[r['p95_mm']*1000 for r in old['links']];newlinks=[r['p95_mm']*1000 for r in new['links']]
    m=arrays(SCAN/'precision_scan_results/merged.npz')
    endpoint=m['overlap_diagnostic_offsets'][-1]@m['basis']*1000
    lines=['# ScanDIC 离线误差整改与验证报告','',
    '日期：2026-09-21。设备已下线；本轮没有连接相机、串口或执行电机运动。原图、原标定、历史结果保留，所有新结果另存。','',
    '## 结论','',
    '**已修复并验证匹配器偏差、双目时序重匹配的噪声放大，以及默认重叠平移链造成的累计漂移。百微米整体偏差需要独立静止参照来分离，不能仅靠算法把每站归零。**','',
    f'- 同点比较，静止 U/V/W 绝对P95从 {numbers(static["common_old"])} 降到 {numbers(static["common_new"])} μm。',
    f'- 14站同点比较为 {numbers(scan["common_old"])} → {numbers(scan["common_new"])} μm：W改善，U整体偏差保留，V未改善。',
    f'- 已知静止假设下，相机位姿拟合的独立空间留出点为 {numbers(pose["corrected_test"])} μm；不是独立加载试验准确度，也不是新的标定。',
    '- 固定站采集几何闸门未通过，毫米/微米量是沿用旧标定的诊断换算；本轮没有把该批数据改标为合格。',
    '- 新默认输出保留表观位移；只有显式提供独立静止锚点才启用相机补偿。硬件稳定性、绝对几何尺度和已知非零位移准确度仍待设备恢复后验证。','',
    '## 数据与统计口径','',
    '|数据|规模|条件|','|---|---|---|',
    '|scan_ref_20260920_184347|14站、28图|基准；每站36891脉冲|',
    '|scan_def_20260920_194648|14站、28图|不加载零位移复扫|',
    '|static_station11_20260920_203311|21组、42图，60.02秒|先到站11，状态0作参考，随后20次固定站比较|','',
    '全部图为4096×3000 uint16，4帧平均，双目曝光50000 μs；16位值/256恢复浮点灰度，不再舍入为8位。绑定标定为20260920_154733，RMS 0.326px，基线228.694mm，夹角24.453°，fx约15681px。','',
    '旧复扫站13曾在ECC中位0.837px处中止；随后固定重复约0.726/0.734/0.676px，正式补采约0.746px通过。所有候选与续采历史仍在 resume_history；最后接受图通过采集闸门不等于全程几何完全稳定。静止站11到站/结束ECC约1.204/1.189px，均未通过0.8px闸门。','',
    'P95指各分量绝对值的95分位，单位μm；保留全局零偏，不减中位、不拟合后冒充原始值。按有效采样点计权，重叠区重复点保留。静止各帧共享参考，扫描各点共享标定，样本数不等于独立重复实验次数；本报告不据此构造虚假的置信区间。','',
    '## 修复内容与已知位移证据','',
    '1. `precise_match.py` 用三次样条插值的局部迭代替代相位相关质心作最终亚像素估计，联合处理亮度增益与偏置；整数/相位结果仅作初值。小数查询坐标不再被当成整数位置。纹理不足、边界不足或迭代不收敛均拒绝。',
    '2. `stereo_temporal.py` 在参考时刻建立一次左右对应，再独立跟踪 L0→L1 与 R0→R1。避免先移动左点、再对当前双目重新匹配，把空间匹配偏差混入时序位移。保留真实全局偏移与三维变形。',
    '3. `precision_scan_pipeline.py` 用精确对应建立三维刚体连接；站间几何坐标变换和位移偏置修正分开。默认 `D == D_raw`，重叠修正仅作为另名诊断输出。',
    '4. `camera_motion.py` 只接受调用方明确指定的静止锚点，分别拟合两相机增量位姿并回到参考世界坐标；固定参考重投影残差作共同项处理。它不改写原标定，不自动假定试件静止。',
    '5. 起点容差由30px收紧至0.3px，4帧平均后进行全分辨率精配；支持已知静止ROI并拒绝单轴不可修正的横向偏差。普通采集粗定位保持原路径。原30px回程检查仅作行程收尾，不能当作复扫对齐精度。硬件部分仅做离线模拟。','',
    f'已知位移基准包含4种纹理、6组位移、9个整数/小数查询位置，以及1.05倍亮度+3灰度偏置。24组内两方法均无失败。真实站11散斑裁剪的非零位移分量P95：旧 {oldrange[0]:.4f}–{oldrange[1]:.4f}px，新 {newrange[0]:.4f}–{newrange[1]:.4f}px。新法最差低平滑合成纹理约0.0124px，不能泛称所有场景0.003px。','',
    '测试还验证：双目非零三维变形被保留；外部锚点补偿后，非锚点区域人为施加的25/−10/60 μm变形保留；缺锚点拒绝补偿；新扫描默认保留200 μm整体偏移；0.15px与0.4px偏移在0.3px起点门槛两侧被正确判别。全部42项回归通过，详见 [tests.txt](tests.txt)。','',
    '当前局部匹配使用平移形函数；本次小运动零试验已验证，不能据此保证大应变、明显子区转动、遮挡或光照形态剧变下的精度。这些场景需扩展形函数并用相应非零形变基准验证。','',
    '![已知位移匹配基准](matcher_benchmark.png)','',
    '## 全量重算：同像素位置公平比较','',
    '|数据与范围|点/状态样本数|旧 U / V / W P95|新 U / V / W P95|','|---|---:|---|---|']
    for label,s in (('14站同像素',scan),('固定站同像素',static)):
        lines.append(f'|{label}|{s["common_new"]["count"]}|{numbers(s["common_old"])}|{numbers(s["common_new"])}|')
    lines+=['','同点比较采用旧处理的同一试件坐标系，避免点集变化或坐标轴小旋转制造改善。扫描共同点29094，静止41262个点/状态样本，全部纳入上表。','',
    f'全部各自有效点：旧扫描{scan["legacy"]["count"]}点 {numbers(scan["legacy"])}，新扫描{scan["raw"]["count"]}点 {numbers(scan["raw"])}；旧静止{static["legacy"]["count"]}样本 {numbers(static["legacy"])}，新静止{static["raw"]["count"]}样本 {numbers(static["raw"])} μm。',
    'V分量上升说明“改用更准确匹配”不保证原始零偏都减小，旧时序重匹配偏差也可能与相对位姿偏差部分相消；现有数据不能逐项唯一分解这种相消。新法W噪声降低和合成真值结果支持修复，但不能把新值一概当作准确真值。','',
    '![同点比较](same_pixel_comparison.png)','',
    '新静止时序及全部数值见 [固定站报告](../static_station11_20260920_203311/precision_static_results/static_validation.md)。该报告采用新拼接坐标轴，数值与上表旧轴A/B对照有微小差异。','',
    f'20个状态共同有效点{series["common_points"]}；几何 qualification=false。原图首末位移小于0.03px，编码器稳定在量化分辨率附近，只能说明短时固定采集明显好于两轮复扫，不能把时间间隔、热漂移与往返运动的贡献完全分开。','',
    '## 剩余整体偏差：相机位姿模型对照','',
    '零试验记录允许作“整个试件静止”的诊断假设。每站将参考左图分成384px交替空间块，一半点拟合，另一半完全不参与拟合；两种模型使用相同训练/留出点。','',
    '|模型|拟合自由度/站|留出点 U / V / W P95 μm|','|---|---:|---|',
    f'|原始表观位移|0|{numbers(pose["raw_test"])}|',
    f'|双目机架整体刚体变化，保持左右R/T固定|6|{numbers(pose["rigid_rig_test"])}|',
    f'|左右相机分别增量位姿|12|{numbers(pose["corrected_test"])}|','',
    '共14879个留出点。整个机架的6自由度模型已经解释绝大部分整体偏差；允许左右相机相对位姿变化后，留出残差进一步减小。相对转角等效估计约0.00074–0.01019°。这些是已知静止假设下的模型等效量，不是测得的某根支架弯曲角；局部标定残差、温漂、焦距变化也可能被位姿参数吸收。不能仅据此认定机架某处松动。',
    '12自由度模型各站留出P95最差约1.12/1.24/4.05 μm。其结果只是现有帧、有效点和固定标定下的空间预测残差；不是另一天、另一试件或外部位移标准的验证。训练点结果与留出点分别存盘，正式默认输出没有采用此假设。','',
    '![相机模型的留出结果](heldout_camera_models.png)','',
    '加载试验中，图像取决于相机与试件的相对位姿：对所有试件点加同一刚体运动、同时相应改变相机位姿，可产生完全相同的投影。没有外部静止锚点或独立位移测量，这6个刚体自由度不能唯一归属于试件或相机。编码器只测电机轴角，不能单独消除传动回差、镜架转动、板体移动和光学几何变化。','',
    '不能把本次每站拟合的零偏数值写成永久补偿表：两次扫描时刻不同，站13还经历续采，既混有空间效应也混有时间变化。永久表需要独立多轮复验。','',
    '## 跨站缩放、散斑与拼接漂移','',
    '轨道与板面几乎平行的判断与图像缩放并不矛盾。此前四图同名点核算显示沿板面位移约57.46/57.50mm、法向仅0.123/0.213mm；左相机斜视约12.6°，同一次平移在其光轴方向投影约12.7mm。左右相机汇聚观察，同一纹理随视角投影可出现左放大、右缩小。因此跨站应采用单应预扭和三维刚体连接，不能由“图像缩放”直接推断滑轨明显不平行。该正常跨站投影变化与同站两次扫描的微小位姿漂移应分开。','',
    '散斑确有大斑、密度不均、局部近饱和等问题；但此前筛到高NCC点后，站内U中位最多只改0.75μm，130–250μm整体偏差仍在。固定站同一块散斑经过软件修复获得更低噪声，也不支持仅靠重喷斑解决整体偏差。噪声、匹配率仍可通过更均匀的哑光随机散斑改善，需控制曝光/标定不变另做对照。','',
    f'参考站间三维连接残差P95：旧{min(oldlinks):.2f}–{max(oldlinks):.2f}μm，新{min(newlinks):.2f}–{max(newlinks):.2f}μm。它们是相邻站形状对应残差，不能当作位移精度；剩余30–40μm提示绝对几何和累积连接仍需外部尺度/形状参照。',
    f'新生产拼接覆盖{new["coverage_mm"][0]:.2f}×{new["coverage_mm"][1]:.2f}mm，共{new["raw"]["count"]}点；新坐标系原始P95为{numbers(new["raw"])}μm。若诊断性地套重叠链，变为{numbers(new["overlap_diagnostic"])}μm，W尾端人为添加{endpoint[2]:.2f}μm。旧链W尾端约−33.73μm，旧W P95从31.74增至46.73μm。',
    '重叠区平移链把局部空间差异逐站积累，接缝连续不能证明绝对准确。故新默认结果不使用该链；示范结果另存 `uvw_overlap_diagnostic`，图中保留失败证据。','',
    '![拼接与累计偏移](stitching_diagnostics.png)','',
    '## 可复现文件与命令','',
    '- [verification.json](verification.json)：98张原图解码与当前SHA256，70张有历史SHA256的图逐一一致；56张扫描图大小/时间戳与旧清单一致；3批绑定标定一致。重算同点统计、每站变换、拼接对应残差与留出统计全部通过。',
    '- `scan_def.../precision_results/`：14站A/B数据；`camera_motion_diagnostic/`：静止假设拟合、分块掩码、留出统计。',
    '- [新扫描报告](../scan_def_20260920_194648/precision_scan_results/report.md)：正式默认表观位移、NPZ、站间连接、运行参数与源码哈希。',
    '- `static_station11.../precision_results/`：20组A/B；`precision_static_results/`：时序报告、共同点立方数组与图。',
    '- 本目录 `matcher_benchmark.json`、`tests.txt`、`verification.json` 和4张PNG记录基准与验证。历史结果只读使用。不同阶段manifest记录运行时源码，最终源码哈希在verification中；后续起点对齐和静止报告封装调整未改变已重算的核心匹配器。','',
    '以下命令均在项目根目录执行，完全离线。已有输出拒绝覆盖时换新目录；A/B缓存要求原图/标定/核心匹配源码身份一致。','',
    '```powershell',
    'python -X utf8 -B -u benchmark_precise.py',
    'python -X utf8 -B -u reprocess_precision.py captures/scan_def_20260920_194648 --mode scan',
    'python -X utf8 -B -u reprocess_precision.py captures/static_station11_20260920_203311 --mode static',
    'python -X utf8 -B -u diagnose_camera_motion.py captures/scan_def_20260920_194648 --known-stationary-zero-test',
    'python -X utf8 -B -u scan_process.py captures/scan_def_20260920_194648 --cache captures/scan_def_20260920_194648/precision_results --overlap-diagnostic --output precision_scan_rerun',
    'python -X utf8 -B -u process_static_station.py captures/static_station11_20260920_203311 --cache captures/static_station11_20260920_203311/precision_results --diagnostic --output precision_static_rerun',
    'python -X utf8 -B -m unittest test_precision test_scan_validation test_scan_pipeline test_static_station -v',
    'python -X utf8 -B -u build_remediation_report.py',
    '```','',
    '## 设备恢复后的验证闭环（本轮未执行）','',
    '1. 保持相机/镜头稳定并充分热稳态，复核机械连接、线缆牵拉和单向接近站位的回差；先在站0、中间、末站分别验证几何闸门。未通过先查明并重标，不放宽阈值来获得结果。',
    '2. 每站视野加入与试件独立、固定于稳定基座的参照。起点原始左图ROI用于回位；处理锚点矩形采用去畸变参考左图坐标，每站至少30个有二维分布的有效点。参照若跟随试件一起运动，就不能分离刚体运动。',
    '3. 同站静止多帧、无加载往返、加载非零位移分开试验，锁定曝光与平均数。重复多轮以检验时间漂移、方向回差和跨站重现性，报告每轮有符号偏差、RMS、绝对P95、有效率和拼接残差，保留失败帧。',
    '4. 用独立可追溯位移参照施加0及正负多个非零位移，例如±50/±100/±200μm，并覆盖面内/离面方向；检验增益、符号、回差和锚点之外真实变形是否保留。滑台命令或其自身编码器不能同时充当独立真值。',
    '5. 用户本次要求先建立基线，未预设合格阈值。后续按实际测量需求与独立参照不确定度确定验收指标；本报告不替用户宣告微米级绝对准确度已合格。','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    checked=audit()
    make_report(checked)
    print('All offline invariants passed. Report:',OUT/'report.md')
