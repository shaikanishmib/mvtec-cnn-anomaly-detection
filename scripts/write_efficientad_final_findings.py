"""Genuine post-freeze failure panels and measured research conclusions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anomaly_detection.efficientad_final import ROOT,REPORT,FREEZE,METRICS,atomic_json
from anomaly_detection.data import CATEGORIES,sha256

OUT=ROOT/'reports/efficientad'
PANELS=ROOT/'outputs/efficientad/final/failure_panels'


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['models']==45
    runs=pd.read_csv(OUT/'all_runs.csv');categories=pd.read_csv(OUT/'all_category_means.csv').set_index('category')
    all_images=[];results={};indices={}
    for row in runs.itertuples():
        r=json.loads((REPORT/f'runs/{row.experiment_id}-result.json').read_text());results[row.experiment_id]=r
        table=pd.read_csv(ROOT/r['images_csv'],float_precision='round_trip');table['seed']=row.seed;table['experiment_id']=row.experiment_id
        table['map_index']=np.arange(len(table));all_images.append(table)
    images=pd.concat(all_images,ignore_index=True)
    dimensions=pd.read_csv(ROOT/'outputs/eda/inventory.csv',usecols=['path','height','width']).set_index('path')
    images['native_height']=images.path.map(dimensions.height);images['native_width']=images.path.map(dimensions.width)
    PANELS.mkdir(parents=True,exist_ok=True)
    examples=[]
    for category in CATEGORIES:
        group=images[images.category.eq(category)]
        chosen=[]
        for outcomes,label in [(['FN'],'false_negative'),(['FP'],'false_positive')]:
            candidates=group[group.outcome.isin(outcomes)].sort_values(['pixel_detected_defect_pixels','score'],ascending=False)
            if len(candidates):chosen.append((label,candidates.iloc[0]))
        for condition,label in [(group.label.eq(1),'small_defect'),(group.label.eq(0),'normal_reference')]:
            candidates=group[condition].sort_values(['defect_fraction','seed','path'])
            if len(candidates):chosen.append((label,candidates.iloc[0]))
        for pattern,role in [('scratch|thread|crack','fine_scratch_or_crack'),('missing|bent|broken','structural_defect')]:
            candidates=group[group.label.eq(1)&group.defect.str.contains(pattern,case=False,regex=True)].sort_values(['predicted','defect_fraction','seed'])
            if len(candidates):chosen.append((role,candidates.iloc[0]))
        seen=set()
        for role,row in chosen:
            unique=(row.experiment_id,row.path)
            if unique in seen:continue
            seen.add(unique);r=results[row.experiment_id];index=int(row.map_index)
            raw=np.load(ROOT/r['raw_branch_maps'],mmap_mode='r')[index].copy();q=np.array(r['training']['map_quantiles'],np.float32)
            maps=[.1*(raw[i]-q[i,0])/(q[i,1]-q[i,0]) for i in range(2)]
            with Image.open(ROOT/'data/mvtec_ad'/row.path) as im:original=np.asarray(im.convert('RGB'))
            size=original.shape[:2]
            native=[F.interpolate(torch.from_numpy(m[None,None]),size=size,mode='bilinear',align_corners=False).numpy()[0,0] for m in maps]
            combined=(native[0]+native[1])*.5
            if isinstance(row['mask'],str):
                with Image.open(ROOT/'data/mvtec_ad'/row['mask']) as im:truth=np.asarray(im)>0
            else:truth=np.zeros(size,bool)
            pixel_predicted=native[0]>r['calibration']['pixel_threshold']
            fig,axes=plt.subplots(1,6,figsize=(18,3.8))
            axes[0].imshow(original);axes[0].set_title('Original')
            for ax,m,title in zip(axes[1:4],[native[0],native[1],combined],['Teacher–student (selected)','Student–AE (not selected)','50/50 combination (not scored)']):
                shown=ax.imshow(m,cmap='inferno');ax.set_title(title,fontsize=9);fig.colorbar(shown,ax=ax,shrink=.65)
            axes[4].imshow(original);axes[4].imshow(np.ma.masked_where(~pixel_predicted,pixel_predicted),cmap='autumn',alpha=.45,vmin=0,vmax=1);axes[4].set_title('Selected-map pixel alarm overlay')
            axes[5].imshow(truth,cmap='gray',vmin=0,vmax=1);axes[5].set_title('Ground truth')
            for ax in axes:ax.axis('off')
            fig.suptitle(f"{category}, seed {row.seed}, {row.outcome}, {role}: {row.path}\nscore {row.score:.6g}; image threshold {r['calibration']['threshold']:.6g}; mask area {row.defect_fraction:.4%}",fontsize=10)
            fig.tight_layout();path=PANELS/f'{category}-s{row.seed}-{role}.png';fig.savefig(path,dpi=110);plt.close(fig)
            zoom_path=None;zoom_box=None
            if truth.any() and role in ('small_defect','fine_scratch_or_crack'):
                ys,xs=np.where(truth);y0=max(0,int(ys.min())-24);y1=min(size[0],int(ys.max())+25)
                x0=max(0,int(xs.min())-24);x1=min(size[1],int(xs.max())+25)
                zoom_box=[y0,y1,x0,x1]
                fig,zaxes=plt.subplots(1,4,figsize=(12,3.5))
                zaxes[0].imshow(original[y0:y1,x0:x1]);zaxes[0].set_title('Actual image crop')
                shown=zaxes[1].imshow(native[0][y0:y1,x0:x1],cmap='inferno');fig.colorbar(shown,ax=zaxes[1],shrink=.7);zaxes[1].set_title('Selected local map')
                zaxes[2].imshow(original[y0:y1,x0:x1]);zaxes[2].imshow(np.ma.masked_where(~pixel_predicted[y0:y1,x0:x1],pixel_predicted[y0:y1,x0:x1]),cmap='autumn',alpha=.45,vmin=0,vmax=1);zaxes[2].set_title('Pixel alarm overlay')
                zaxes[3].imshow(truth[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1);zaxes[3].set_title('Actual ground truth crop')
                for ax in zaxes:ax.axis('off')
                fig.suptitle(f'{category}, seed {row.seed}, {row.outcome}; post-evaluation GT-bounding-box zoom, no scoring change')
                fig.tight_layout();zoom_path=PANELS/f'{category}-s{row.seed}-{role}-zoom.png';fig.savefig(zoom_path,dpi=130);plt.close(fig)
            examples.append(dict(category=category,seed=int(row.seed),role=role,path=row.path,outcome=row.outcome,score=float(row.score),
                threshold=r['calibration']['threshold'],defect_fraction=float(row.defect_fraction),pixel_overlap=int(row.pixel_detected_defect_pixels),
                predicted_background_pixels=int(row.pixel_predicted_pixels-row.pixel_detected_defect_pixels),panel=path.relative_to(ROOT).as_posix(),
                panel_sha256=sha256(path),zoom_panel=zoom_path.relative_to(ROOT).as_posix() if zoom_path else None,
                zoom_sha256=sha256(zoom_path) if zoom_path else None,zoom_native_bbox=json.dumps(zoom_box) if zoom_box else None,
                map_max_at_native_border=bool(row.native_peak_y<16 or row.native_peak_x<16 or row.native_peak_y>=size[0]-16 or row.native_peak_x>=size[1]-16)))
    examples=pd.DataFrame(examples);examples.to_csv(OUT/'failure_examples.csv',index=False,float_format='%.17g',lineterminator='\n')
    assert set(examples.category)==set(CATEGORIES)
    compact=[]
    for category in CATEGORIES:
        group=images[images.category.eq(category)];normal=group[group.label.eq(0)];defect=group[group.label.eq(1)]
        compact.append(dict(category=category,seed_evaluations=len(group),false_positives=int((group.outcome=='FP').sum()),false_negatives=int((group.outcome=='FN').sum()),
            fn_with_any_localized_pixel_alarm=int(((group.outcome=='FN')&(group.pixel_detected_defect_pixels>0)).sum()),
            tiny_defects=int((defect.defect_fraction<=.001).sum()),tiny_defect_recall=float(defect.loc[defect.defect_fraction<=.001,'predicted'].mean()) if (defect.defect_fraction<=.001).any() else None,
            normal_peak_border_fraction=float(((normal.native_peak_y<16)|(normal.native_peak_x<16)|
                (normal.native_peak_y>=normal.native_height-16)|(normal.native_peak_x>=normal.native_width-16)).mean())))
    behavior=pd.DataFrame(compact);behavior.to_csv(OUT/'category_failure_summary.csv',index=False,lineterminator='\n')
    macro=summary['macro'];comparison=pd.read_csv(OUT/'model_macro_comparison.csv',index_col=0).T
    costs=pd.read_csv(OUT/'model_cost_comparison.csv')
    lines=['# EfficientAD-S reduced-budget final benchmark','',
        'All 45 frozen category-seed models completed. The initial pilot and four development fits remain separate. No architecture, hyperparameter, score or threshold policy changed using test results.','',
        f"Frozen identity: `{FREEZE}`. Nine full epochs, batch one, deterministic FP32, pinned PDN teacher, dual-head student and AE objectives, no external-image penalty. The selected score is normalized local-map maximum; global/combined maps are retained for analysis only. This is not a paper-equivalent reproduction.",'',
        '## All nine macro metrics','', '| Metric | Mean | Seed SD | Conditional image-bootstrap 95% interval |','|---|---:|---:|---|']
    for k in METRICS:
        value=macro[k];ci=value['image_bootstrap_95ci'];lines.append(f"| {k} | {value['mean']:.6f} | {value['seed_std']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines.extend(['','Image intervals use exact scores with shared whole-image resamples across seeds/models. Pixel intervals use 4,096 signed monotone bins; pixel point estimates are exact native-resolution metrics. Quantization bias is retained in summary.json. These intervals condition on fitted models and thresholds and do not include fitting/calibration uncertainty. Shared metric conventions set precision to zero for zero-alert runs; those cases are explicitly counted rather than treated as a measured positive predictive probability.','',
        '## Category behavior and failure evidence','',
        '| Category | Image AUROC | Pixel AUROC | Recall | FPR | FP evaluations | FN evaluations | FN with pixel overlap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for category in CATEGORIES:
        r=categories.loc[category];b=behavior[behavior.category.eq(category)].iloc[0]
        lines.append(f"| {category} | {r.image_auroc_mean:.4f} | {r.pixel_auroc_mean:.4f} | {r.recall_mean:.4f} | {r.fpr_mean:.4f} | {b.false_positives} | {b.false_negatives} | {b.fn_with_any_localized_pixel_alarm} |")
    lines.extend(['','Counts are seed-evaluations of the same official images, not independent new images. All 15 categories have genuine original/map/mask panels under ignored outputs/efficientad/final/failure_panels. The committed failure_examples.csv identifies every source and outcome. Where a category has no false positive or false negative, no such example is fabricated.','',
        f"Observed: {summary['fn_with_pixel_overlap']} image false-negative evaluations still have at least one pixel alarm overlapping the true defect at the auxiliary normal-calibrated pixel threshold. This demonstrates image aggregation/operating-point misses, not that every localized response is accurate. See mask area strata and per-defect tables for fine scratches and small defects.",'',
        'Observed texture alarms, leather/carpet results, screw detection, tiny-mask recall and normal peak locations are quantified in the tables. Pixel background alarm area and genuine overlays expose localization spread. Heatmaps have per-panel signed-value colorbars; brightness alone is not comparable across panels. A spatial halo caused by receptive fields or padding is a hypothesis unless established by the displayed case; it is not inferred from a scalar AUROC. Structural classes are reported by the actual dataset defect names. MVTec AD is not MVTec LOCO and cannot establish comprehensive logical-anomaly competence.','',
        '## Calibration','',
        f"Mean observed normal calibration FPR {summary['calibration_fpr_mean']:.4f}; test macro FPR {summary['test_fpr_macro']:.4f}. {summary['calibration_runs_with_ties']} of 45 calibration score sets contain ties. Thresholds use higher 95th percentile and strict greater, always from reserved clean normals. Tied-score Beta intervals are inapplicable; zero-width bootstraps do not prove certainty. Per-run sample sizes, thresholds, bootstrap/leave-one-out sensitivity and order-statistic qualifications are saved.",'',
        'Bottle response floors are assessed from saved per-seed calibration distributions and native peak positions, without cropping borders or changing score selection after test. The precision correction from development is preserved: exact float32 values are promoted before 17-digit CSV and parsed with round-trip precision.','',
        '## Six-model comparison','', '| Model | '+' | '.join(METRICS)+' |','|---|'+'---:|'*len(METRICS)])
    for name,row in comparison.iterrows():lines.append('| '+name+' | '+' | '.join(f'{row[k]:.4f}' for k in METRICS)+' |')
    lines.extend(['','All previous model entries come from their exact stored summaries and matching 15-category/three-seed tables; none were changed. Per-category joins and paired differences are retained. Paired exact-image bootstrap intervals cover six image metrics; paired category resampling covers all nine primary metrics. These are different uncertainty targets. No multiplicity correction is applied, and small macro differences are not evidence of general superiority.','',
        '| Model | Fit/train s | Eval s | Model p50/p95 ms | E2E p50/p95 ms | Peak train VRAM GiB | Peak eval RAM GiB | Category checkpoint bytes | Including shared trunk bytes |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|'])
    for row in costs.itertuples():
        ram=f'{row.peak_cpu_rss_gib:.3f}' if pd.notna(row.peak_cpu_rss_gib) else 'not recorded'
        lines.append(f'| {row.model} | {row.training_fitting_seconds:.2f} | {row.evaluation_seconds:.2f} | {row.model_p50_ms:.2f}/{row.model_p95_ms:.2f} | {row.end_to_end_p50_ms:.2f}/{row.end_to_end_p95_ms:.2f} | {row.peak_training_vram_gib:.3f} | {ram} | {row.max_checkpoint_bytes} | {row.single_category_deployment_bytes} |')
    lines.extend(['','Costs are recorded local fitting/evaluation effort, not total pretraining effort. Shared ResNet18 layer1-3 trunk size comes from the exact recorded 11,179,957-byte export; PaDiM reuses the same verified PatchExtractor trunk. The trunk is shared across categories/seeds, not duplicated 45 times. Original weight downloads and persistent feature caches are separate; feature-cache bytes are retained in the cost CSV and are not deployment-model size. Earlier latency scopes differ and CUDA scheduling/power state varied across sessions. Basic CNN/AE evaluation RAM was not recorded. EfficientAD timings conservatively include both raw maps even though only the local map is scored. Per-run percentiles are averaged, not pooled into a new latency distribution.','',
        '## Actual resources and budget','',
        f"Training {summary['total_training_seconds']:.2f} s; calibration {summary['total_calibration_seconds']:.2f} s; evaluation {summary['total_evaluation_seconds']:.2f} s; run pipelines {summary['total_run_pipeline_seconds']/3600:.3f} hours. The 2.68-hour training and 10.16-hour conservative forecasts remain historical projections, not substituted for these measurements.",'',
        f"Final-checkpoint serialization and replay verification total {summary['total_checkpoint_serialization_verification_seconds']:.2f} s; teacher preparation {summary['total_preparation_seconds']:.2f} s; development-map scaling {summary['total_map_normalization_seconds']:.2f} s. Per-epoch progress snapshot overhead is included in pipeline wall time but is not separately isolated.",'',
        f"Peak training allocated/reserved VRAM {summary['peak_training_gpu_allocated_gib']:.3f}/{summary['peak_training_gpu_reserved_gib']:.3f} GiB; peak inference allocated/reserved {summary['peak_inference_allocated_gib']:.3f}/{summary['peak_inference_reserved_gib']:.3f} GiB; sampled worker peak RSS {summary['peak_evaluation_cpu_rss_gib']:.3f} GiB. Maximum checkpoint {summary['max_checkpoint_bytes']:,} bytes includes teacher, student and AE. Manager process and driver-reserved memory are outside worker RSS/PyTorch allocator scopes.",'',
        '## Limitations and conclusion','',
        f"The fixed epoch budget produces {summary['min_optimizer_steps_per_model']}–{summary['max_optimizer_steps_per_model']} optimizer steps per model ({summary['total_optimizer_steps']:,} overall), so category differences also involve different update counts. This is disclosed, not retrospectively equalized.",'',
        'The pinned third-party teacher has repository Apache-2.0 licensing but no independently verified separate weight license/training log. No legitimate external-image penalty input was available. Nine epochs, one-seed/two-category optimistic proxy selection, full-image max floors, small normal calibration sets and exclusion of the global map limit conclusions. Exact test results characterize this reduced variant only. Real-data panels and all failed/interrupted attempts are preserved. Notebook 07 — Final Model Comparison is the next milestone and has not started.',''])
    (ROOT/'docs/efficientad_final_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved',len(examples),'genuine panels across 15 categories')


if __name__=='__main__':main()
