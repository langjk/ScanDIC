"""零位移基线的科学图与可追溯Markdown报告。"""
import numpy as np


def plot_results(out, xy, raw, corrected, station, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(16,8),layout='constrained')
    for i, name in enumerate('UVW'):
        vmax = max(1, np.percentile(np.abs(np.r_[raw[:,i],corrected[:,i]])*1000,98))
        for j, (label, D) in enumerate((('Raw',raw),('Overlap corrected; station 0 anchor',corrected))):
            ax = axes[i,j]
            p = ax.scatter(xy[:,0],xy[:,1],c=D[:,i]*1000,s=2,cmap='coolwarm',vmin=-vmax,vmax=vmax)
            ax.set(title=f'{label}: {name}',xlabel='S (mm)',ylabel='T (mm)',aspect='equal')
            fig.colorbar(p,ax=ax,label='um',shrink=0.8)
    fig.savefig(out/'zero_displacement_fields.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,name in enumerate('UVW'):
        for label, D in (('raw',raw),('corrected',corrected)):
            med = [np.median(D[station==k,i])*1000 for k in range(len(report['stations']))]
            axes[i].plot(range(len(med)),med,'o-',label=label)
        axes[i].axhline(0,color='gray',lw=0.7)
        axes[i].set(title=name,xlabel='Station',ylabel='Median displacement (um)')
        axes[i].legend()
    fig.savefig(out/'station_bias.png',dpi=160)
    plt.close(fig)


def write_report(out, report):
    lines = ['# 不加载复扫：零位移实测基线','',
             '本次不预设合格阈值；以下为一次基准/复扫的重复性测量，不代表已知位移准确度。', '',
             f"- 基准：`{report['ref_dir']}`",f"- 复扫：`{report['def_dir']}`",
             f"- 有效点 {report['raw']['count']}；实测覆盖 {report['coverage_mm'][0]:.1f} × {report['coverage_mm'][1]:.1f} mm（扫描视场覆盖的条带）。",
             '- 位移单位 μm；统计按各站有效点合并，重叠点保留，因此不是面积均匀加权。',
             '- U/V 为共同试件平面轴，U 接近站0相机+x；W 朝相机。原始场保留相机复位造成的伪位移。',
             '- 修正仅利用相邻站重叠位移差作平移链，站0偏差保留。修正后接缝改善属于拟合一致性，不能视为独立准确度验证。','',
             '|结果|分量|中位|均值|RMS|绝对值P95|','|---|---|---:|---:|---:|---:|']
    for kind,label in (('raw','原始'),('corrected','重叠修正')):
        for name in 'UVW':
            s=report[kind][name]
            lines.append(f"|{label}|{name}|{s['median_um']:+.2f}|{s['mean_um']:+.2f}|{s['rms_um']:.2f}|{s['p95_abs_um']:.2f}|")
    lines += ['', '## 逐站原始偏差', '', '|站|有效点|U中位|V中位|W中位|U绝对P95|V绝对P95|W绝对P95|',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,s in enumerate(report['stations']):
        r=s['raw']
        vals=[r[a]['median_um'] for a in 'UVW']+[r[a]['p95_abs_um'] for a in 'UVW']
        lines.append(f"|{k}|{r['count']}|"+'|'.join(f'{v:.2f}' for v in vals)+'|')
    lines += ['', '## 站间几何与接缝', '', '|连接|刚体内点/总数|几何残差中位/P95 mm|原始接缝P95 μm|修正接缝P95 μm|',
              '|---|---:|---:|---:|---:|']
    for link,seam in zip(report['links'],report['seams']):
        lines.append(f"|{seam['stations'][0]}→{seam['stations'][1]}|{link['inliers']}/{link['count']}|{link['median_mm']:.4f}/{link['p95_mm']:.4f}|{seam['raw_p95_norm_um']:.2f}|{seam['corrected_p95_norm_um']:.2f}|")
    lines += ['', '## 方法与边界', '',
              f"- 参数：全分辨率；子区 {report['parameters']['subset']} px；采样步长 {report['parameters']['step']} px；搜索半径 {report['parameters']['search']} px；ZNCC≥0.7；极线距离≤1.5px。",
              '- 站间先用单应补偿透视，再匹配局部残差，映回原坐标并在下一站精确三角化，未使用最近网格点代替三维对应。',
              '- 刚体拟合 RANSAC 阈值0.15mm；失败或缺站立即中止，不复用前站变换。接缝最近点半径2mm；其误差也含不同采样位置的场差。',
              '- 未按位移大小删除离群点；原始与修正统计均保留所有通过图像/几何检查的有限三维点。',
              '- 局部相位相关仍有纹理相关的亚像素偏差；该次实验不能分离运动回位、相机姿态/标定漂移与匹配器误差。',
              '- 输出包含逐站对应、三维点、原始位移、站间匹配、累计变换、修正向量、原始及修正云图，可追溯复算。', '',
              '![位移场](zero_displacement_fields.png)', '', '![逐站偏差](station_bias.png)', '']
    (out/'zero_validation.md').write_text('\n'.join(lines),encoding='utf-8')
