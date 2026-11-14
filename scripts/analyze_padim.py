"""Retrospective failure descriptions; never feed these data into selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from anomaly_detection.data import CATEGORIES,autodiscover_data_root,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/padim';data=autodiscover_data_root(ROOT)
    tables=[];examples=[];areas={}
    for category in CATEGORIES:
        for seed in (2026,2027,2028):
            identifier=f'05-test-{category}-s{seed}-v1'
            result=json.loads((folder/f'evaluation/{identifier}-result.json').read_text())
            table=pd.read_csv(folder/f'evaluation/{identifier}-images.csv')
            table['seed']=seed;table['image_threshold']=result['calibration']['image_threshold']
            table['pixel_threshold']=result['calibration']['pixel_threshold'];table['maps_path']=result['local_maps']
            table['row_index']=np.arange(len(table));table['score_minus_threshold']=table.score-table.image_threshold
            fractions=[]
            for row in table.itertuples():
                if isinstance(row.mask,str):
                    if row.mask not in areas:
                        with Image.open(data/row.mask) as im:areas[row.mask]=float((np.asarray(im)>0).mean())
                    fractions.append(areas[row.mask])
                else:fractions.append(0.)
            table['defect_fraction']=fractions
            tables.append(table)
            if seed==2026:
                for outcome in ('FP','FN','TP','TN'):
                    subset=table[table.outcome.eq(outcome)]
                    if len(subset):
                        example=subset.sort_values('score_minus_threshold',ascending=outcome in ('FN','TN')).iloc[0].to_dict()
                        examples.append(example)
    table=pd.concat(tables,ignore_index=True)
    table.to_csv(folder/'all_image_diagnostics.csv',index=False,lineterminator='\n')
    pd.DataFrame(examples).to_csv(folder/'failure_examples.csv',index=False,lineterminator='\n')
    defects=table[table.label.eq(1)].copy()
    defects['size_stratum']=pd.cut(defects.defect_fraction,[0,.001,.01,.05,1],labels=['<=0.1%','0.1-1%','1-5%','>5%'],include_lowest=True)
    strata=defects.groupby('size_stratum',observed=False).agg(images_across_seeds=('predicted','size'),detected=('predicted','sum'),mean_score=('score','mean'))
    strata['recall']=strata.detected/strata.images_across_seeds
    strata.to_csv(folder/'small_defect_summary.csv',lineterminator='\n')
    defects.groupby(['category','defect']).agg(images_across_seeds=('predicted','size'),recall=('predicted','mean'),
        mean_defect_fraction=('defect_fraction','mean')).to_csv(folder/'defect_type_size_summary.csv',lineterminator='\n')
    print(strata.to_string())


if __name__=='__main__':main()
