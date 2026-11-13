"""Retrospective native defect-size and fitted-bank source coverage analysis."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.data import CATEGORIES,autodiscover_data_root
torch.set_num_threads(4)
output=ROOT/'reports/patchcore';frozen=json.loads((ROOT/'configs/patchcore_frozen.json').read_text())
data=autodiscover_data_root(ROOT)
coverage=[];sizes=[]
for seed in frozen['seeds']:
    for category in CATEGORIES:
        identifier=f"03-final-{frozen['selected_variant']}-{category}-s{seed}-v1"
        run=json.loads((output/f'runs/{identifier}-result.json').read_text())
        bank=torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True)
        indices=bank['indices'].numpy()
        counts=np.bincount(indices//1024,minlength=run['fitting_normals'])
        coverage.append(dict(category=category,seed=seed,bank_size=len(indices),unique_patch_indices=len(np.unique(indices)),
            fitting_images=run['fitting_normals'],represented_fitting_images=int((counts>0).sum()),
            fraction_fitting_images_represented=float((counts>0).mean()),
            max_bank_fraction_from_one_image=float(counts.max()/counts.sum()),
            candidate_projected_radius=run['candidate_coverage']['projected_radius']))
        if seed!=2026:continue
        evaluation=f"03-test-{frozen['selected_variant']}-{category}-s{seed}-v1"
        images=pd.read_csv(output/f'evaluation/{evaluation}-images.csv')
        for row in images[images.label.eq(1)].itertuples():
            with Image.open(data/row.mask) as image:mask=np.asarray(image)>0
            sizes.append(dict(category=category,path=row.path,defect=row.defect,native_mask_pixels=int(mask.sum()),
                native_mask_fraction=float(mask.mean()),recalled=bool(row.predicted),score=row.score))
pd.DataFrame(coverage).to_csv(output/'memory_coverage.csv',index=False,lineterminator='\n')
table=pd.DataFrame(sizes)
table['size_group']=pd.cut(table.native_mask_fraction,[0,.001,.01,1],labels=['at most 0.1%','0.1% to 1%','above 1%'],include_lowest=True)
table.to_csv(output/'defect_size_analysis.csv',index=False,lineterminator='\n')
table.groupby('size_group',observed=True).agg(images=('path','size'),recall=('recalled','mean'),median_mask_fraction=('native_mask_fraction','median')).to_csv(output/'small_defect_summary.csv',lineterminator='\n')
table.groupby(['category','defect']).agg(images=('path','size'),recall=('recalled','mean'),median_mask_fraction=('native_mask_fraction','median')).to_csv(output/'defect_type_size_summary.csv',lineterminator='\n')
print('Retained all 45 bank coverage summaries and seed-2026 native defect sizes')
