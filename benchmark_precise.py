"""Known subpixel shifts on multiple textures, including real saved speckles. No hardware."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.ndimage import fourier_shift
from precise_match import SplineImage,PreciseMatcher
from offline_dic import SubsetMatcher


def main():
    rng=np.random.default_rng(19)
    textures={f'gaussian_sigma_{s}':cv2.GaussianBlur(rng.normal(128,40,(384,384)).astype(np.float32),(0,0),s) for s in (.8,1.2,2.)}
    path=Path('captures/scan_ref_20260920_184347/station_11_L.png')
    raw=cv2.imread(str(path),-1).astype(np.float32)/256
    textures['real_station11']=raw[1200:1584,1800:2184].copy()
    old=SubsetMatcher(half=32,search=10)
    new=PreciseMatcher(half=32,search=10)
    reports=[]
    for name,a in textures.items():
        before=a.copy();aa=SplineImage(a)
        for dx,dy in ((0,0),(.25,-.4),(.5,.5),(3.25,2.25),(-3.25,-2.25),(6.8,-.15)):
            b=np.fft.ifftn(fourier_shift(np.fft.fftn(a),(dy,dx))).real.astype(np.float32)
            # Known photometric gain/offset as well as geometric shift.
            b=b*1.05+3
            bb=SplineImage(b);errors={'phase':[],'spline':[]};failed={'phase':0,'spline':0}
            for x,y in ((96,96),(192,96),(288,96),(96,192),(192,192),(288,192),(96.37,288.21),(192.37,288.21),(288.37,288.21)):
                for method,m in (('phase',old.match(a,b,x,y)),('spline',new.match(aa,bb,x,y))):
                    if m is None:
                        failed[method]+=1
                    else:
                        errors[method].extend(np.abs(np.array(m[:2])-[dx,dy]))
            reports.append(dict(texture=name,shift=[dx,dy],p95_component_error_px={k:float(np.percentile(v,95)) if v else None for k,v in errors.items()},failed=failed))
        np.testing.assert_array_equal(a,before)
    out=Path('captures/offline_error_remediation');out.mkdir(exist_ok=True)
    (out/'matcher_benchmark.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    main()
