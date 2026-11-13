"""Post-freeze native-mask/score analysis and deterministic local panel choices."""
import json,sys,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from scipy.stats import beta,binomtest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from anomaly_detection.data import CATEGORIES,autodiscover_data_root,sha256
from anomaly_detection.cnn_research import read_partition
torch.set_num_threads(4)
out=ROOT/'reports/supersimplenet';data=autodiscover_data_root(ROOT)
frozen=json.loads((ROOT/'configs/supersimplenet_frozen.json').read_text())
assert frozen['seeds'][0]==2026
sizes=[];examples=[];stability=[]
def calibration_diagnostic(result,normals,category,seed):
    calibration=np.array(result['calibration']['calibration_scores']);n=len(calibration);rank=math.ceil(.95*(n-1))
    ci=beta.ppf([.025,.975],n-rank,rank+1).tolist()
    test_fpr_ci=binomtest(result['fp'],result['fp']+result['tn']).proportion_ci(method='wilson')
    return dict(category=category,seed=seed,calibration_images=n,target_fpr=.05,image_threshold=result['calibration']['image_threshold'],
        calibration_empirical_fpr=float((calibration>result['calibration']['image_threshold']).mean()),test_fpr=result['fpr'],
        test_fpr_bootstrap_low=result['uncertainty']['95ci']['fpr'][0],test_fpr_bootstrap_high=result['uncertainty']['95ci']['fpr'][1],
        test_fpr_wilson_low=float(test_fpr_ci.low),test_fpr_wilson_high=float(test_fpr_ci.high),
        calibration_iqr=float(np.quantile(calibration,.75)-np.quantile(calibration,.25)),
        test_normal_iqr=float(normals.score.quantile(.75)-normals.score.quantile(.25)),
        calibration_std=float(calibration.std()),test_normal_mean_minus_calibration_mean=float(normals.score.mean()-calibration.mean()),
        iid_continuous_order_stat_expected_fpr=(n-rank)/(n+1),iid_continuous_order_stat_fpr_low=ci[0],iid_continuous_order_stat_fpr_high=ci[1],
        threshold_bootstrap_low=result['calibration']['image_threshold_bootstrap_95ci'][0],threshold_bootstrap_high=result['calibration']['image_threshold_bootstrap_95ci'][1])
for category in CATEGORIES:
    identifier=f'04-test-{category}-s2026-v1';result=json.loads((out/f'evaluation/{identifier}-result.json').read_text())
    table=pd.read_csv(out/f'evaluation/{identifier}-images.csv');rows=read_partition('test',category)
    maps=np.load(ROOT/result['local_maps'],mmap_mode='r');assert len(maps)==len(table)
    candidates=[]
    for outcome in ['FP','FN','TP']:
        subset=table[table.outcome.eq(outcome)].sort_values('path')
        if len(subset):candidates.append((subset.index[0],outcome+' first path'))
    normals=table[table.label.eq(0)]
    if len(normals):candidates.append((normals.score.idxmax(),'highest-score normal'))
    scratches=table[table.defect.str.contains('scratch')].sort_values('path')
    if len(scratches):candidates.append((scratches.index[0],'first scratch'))
    chosen=dict(candidates)
    for i,row in table.iterrows():
        if row.label:
            mask_path=data/row['mask'];assert sha256(mask_path)==rows.mask_sha256.iloc[i]
            with Image.open(mask_path) as image:truth=np.asarray(image)>0
            values=dict(category=category,path=row.path,defect=row.defect,native_mask_pixels=int(truth.sum()),
                native_mask_fraction=float(truth.mean()),recalled=bool(row.predicted),score=float(row.score))
            sizes.append(values)
        elif i in chosen:
            with Image.open(data/row.path) as image:truth=np.zeros((image.height,image.width),bool)
        else:continue
        if i in chosen:
            native=torch.nn.functional.interpolate(torch.from_numpy(np.array(maps[i]))[None,None],size=truth.shape,mode='bilinear',align_corners=False).numpy()[0,0]
            flat=native.ravel();k=math.ceil(flat.size*.01);top=np.argpartition(flat,-k)[-k:]
            examples.append(dict(category=category,path=row.path,mask=None if not row.label else row['mask'],
                defect=row.defect,outcome=row.outcome,reason=chosen[i],row_index=int(i),score=float(row.score),
                image_threshold=result['calibration']['image_threshold'],pixel_threshold=result['calibration']['pixel_threshold'],
                score_minus_threshold=float(row.score-result['calibration']['image_threshold']),
                native_mask_fraction=float(truth.mean()),top1pct_pixel_defect_overlap=float(truth.ravel()[top].mean()),
                predicted_native_pixel_fraction=float((native>result['calibration']['pixel_threshold']).mean()),
                defect_pixel_recall=float((native[truth]>result['calibration']['pixel_threshold']).mean()) if truth.any() else None,
                mean_defect_minus_background_logit=float(native[truth].mean()-native[~truth].mean()) if truth.any() and (~truth).any() else None,
                map_min=float(native.min()),map_max=float(native.max()),maps_path=result['local_maps']))
    stability.append(calibration_diagnostic(result,normals,category,2026))

for seed in frozen['seeds'][1:]:
    for category in CATEGORIES:
        identifier=f'04-test-{category}-s{seed}-v1'
        result=json.loads((out/f'evaluation/{identifier}-result.json').read_text())
        images=pd.read_csv(out/f'evaluation/{identifier}-images.csv')
        stability.append(calibration_diagnostic(result,images[images.label.eq(0)],category,seed))
table=pd.DataFrame(sizes);table['size_group']=pd.cut(table.native_mask_fraction,[0,.001,.01,1],labels=['at most 0.1%','0.1% to 1%','above 1%'],include_lowest=True)
table.to_csv(out/'defect_size_analysis.csv',index=False,lineterminator='\n')
table.groupby('size_group',observed=True).agg(images=('path','size'),recall=('recalled','mean'),median_mask_fraction=('native_mask_fraction','median')).to_csv(out/'small_defect_summary.csv',lineterminator='\n')
table.groupby(['category','defect']).agg(images=('path','size'),recall=('recalled','mean'),median_mask_fraction=('native_mask_fraction','median')).to_csv(out/'defect_type_size_summary.csv',lineterminator='\n')
pd.DataFrame(examples).to_csv(out/'failure_examples.csv',index=False,lineterminator='\n')
pd.DataFrame(stability).to_csv(out/'calibration_stability.csv',index=False,lineterminator='\n')
print('Retained native defect-size summaries, normal threshold diagnostics and deterministic panel choices for15 categories; no model changes')
