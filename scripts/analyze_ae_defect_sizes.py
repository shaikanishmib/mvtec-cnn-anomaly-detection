"""Descriptive post-freeze defect-size analysis; cannot select policy."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.data import CATEGORIES,autodiscover_data_root

output=ROOT/'reports/autoencoder'
frozen=json.loads((ROOT/'configs/autoencoder_frozen.json').read_text())
data=autodiscover_data_root(ROOT)
rows=[]
for category in CATEGORIES:
    identifier=f"02-test-{frozen['selected_variant']}-{category}-s2026-v1"
    table=pd.read_csv(output/f'evaluation/{identifier}-images.csv')
    for row in table[table.label.eq(1)].itertuples():
        with Image.open(data/row.mask) as image:mask=np.asarray(image)>0
        rows.append(dict(category=category,path=row.path,defect=row.defect,
            native_mask_pixels=int(mask.sum()),native_mask_fraction=float(mask.mean()),
            recalled=bool(row.predicted),score=row.score))
table=pd.DataFrame(rows)
# Descriptive size bins chosen after freeze; never used to tune thresholds.
table['size_group']=pd.cut(table.native_mask_fraction,[0,.001,.01,1],
    labels=['at most 0.1%','0.1% to 1%','above 1%'],include_lowest=True)
table.to_csv(output/'defect_size_analysis.csv',index=False,lineterminator='\n')
summary=table.groupby('size_group',observed=True).agg(images=('path','size'),recall=('recalled','mean'),
    median_mask_fraction=('native_mask_fraction','median'))
summary.to_csv(output/'small_defect_summary.csv',lineterminator='\n')
table.groupby(['category','defect']).agg(images=('path','size'),recall=('recalled','mean'),
    median_mask_fraction=('native_mask_fraction','median')).to_csv(output/'defect_type_size_summary.csv',lineterminator='\n')
print(summary.to_string())
