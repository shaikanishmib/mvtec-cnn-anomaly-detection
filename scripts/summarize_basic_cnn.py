"""Verify retained artifacts and aggregate frozen results without new inference."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from anomaly_detection.cnn_research import ROOT
from anomaly_detection.data import CATEGORIES,TEXTURES,sha256

METRICS=["image_auroc","image_ap","pixel_auroc","pixel_ap","au_pro","recall","fpr","precision","f1"]


def main():
    freeze=json.loads((ROOT/"configs/basic_cnn_frozen.json").read_text())
    output=ROOT/"reports/basic_cnn"
    rows=[];defects=[];artifacts=[];bootstrap={key:[] for key in METRICS}
    quantization_bias={key:[] for key in ("pixel_auroc","pixel_ap","au_pro")}
    for seed in freeze["seeds"]:
        seed_boot={key:[] for key in METRICS}
        for category in CATEGORIES:
            identifier=f"01-test-{freeze['selected_variant']}-{category}-s{seed}-v1"
            path=output/f"evaluation/{identifier}-result.json"
            result=json.loads(path.read_text())
            for key in quantization_bias:
                quantization_bias[key].append(result["bootstrap_binned_pixel_point_estimates"][key]-result[key])
            assert result["freeze_sha256"]==sha256(ROOT/"configs/basic_cnn_frozen.json")
            run=json.loads((output/f"runs/{result['training_experiment']}-result.json").read_text())
            assert sha256(ROOT/run["checkpoint"])==run["checkpoint_sha256"]==result["checkpoint_sha256"]
            assert sha256(ROOT/run["history"])==run["history_sha256"]
            assert sha256(ROOT/result["local_maps"])==result["maps_sha256"]
            assert result["category"]==category and result["seed"]==seed
            row={key:result[key] for key in METRICS+["tn","fp","fn","tp","training_seconds","parameters","checkpoint_bytes"]}
            row.update(category=category,seed=seed,experiment_id=identifier,
                       kind="texture" if category in TEXTURES else "object",
                       train_peak_vram_gib=result["training_peak_allocated_bytes"]/2**30,
                       image_threshold=result["calibration"]["image_threshold"],
                       pixel_threshold=result["calibration"]["pixel_threshold"],
                       calibration_images=result["calibration"]["images"],
                       model_p50_ms=result["latency"]["model"]["p50_ms"],model_p95_ms=result["latency"]["model"]["p95_ms"],
                       end_to_end_p50_ms=result["latency"]["end_to_end"]["p50_ms"],end_to_end_p95_ms=result["latency"]["end_to_end"]["p95_ms"],
                       q99_recall=result["secondary_image_q99"]["recall"],q99_fpr=result["secondary_image_q99"]["fpr"],
                       pixel_recall=result["pixel_operating_point"]["recall"],pixel_fpr=result["pixel_operating_point"]["fpr"],
                       pixel_precision=result["pixel_operating_point"]["precision"],pixel_f1=result["pixel_operating_point"]["f1"])
            rows.append(row);defects.extend(result["per_defect"])
            rows[-1]["precision_defined"]=bool(result["tp"]+result["fp"])
            rows[-1]["pixel_precision_defined"]=bool(result["pixel_operating_point"]["tp"]+result["pixel_operating_point"]["fp"])
            local=(ROOT/result["local_maps"]).parent
            for file in [path,ROOT/run["checkpoint"],ROOT/run["history"],ROOT/result["local_maps"],
                         local/"bootstrap_vectors.npz",local/"native_image_histograms.npz",local/"latency_ms.csv"]:
                artifacts.append({"path":file.relative_to(ROOT).as_posix(),"sha256":sha256(file),"bytes":file.stat().st_size})
            image_table=pd.read_csv(output/f"evaluation/{identifier}-images.csv")
            rows[-1]["unique_image_scores"]=int(image_table.score.nunique())
            rows[-1]["fraction_image_scores_equal_one"]=float(image_table.score.eq(1.).mean())
            rows[-1]["normal_test_score_max"]=float(image_table.loc[image_table.label.eq(0),"score"].max())
            y=image_table.label.to_numpy();n=len(y)
            rng=np.random.default_rng(2026+CATEGORIES.index(category))
            weights=rng.multinomial(n,np.full(n,1/n),size=2000)
            positive=weights@y;valid=(positive>0)&(positive<n)
            with np.load(local/"bootstrap_vectors.npz") as vectors:
                for key in METRICS:
                    values=vectors[key]
                    assert len(values)==valid.sum()
                    np.testing.assert_allclose(np.quantile(values,[.025,.975]),result["uncertainty"]["95ci"][key])
                    full=np.full(2000,np.nan);full[valid]=values
                    seed_boot[key].append(full)
        for key in METRICS:bootstrap[key].append(seed_boot[key])
    table=pd.DataFrame(rows)
    assert len(table)==45 and table.groupby("category").size().eq(3).all()
    table.to_csv(output/"all_runs.csv",index=False,lineterminator="\n")
    group=table.groupby("category").agg({k:["mean","std"] for k in METRICS})
    group.columns=["_".join(c) for c in group.columns]
    group.reindex(CATEGORIES).to_csv(output/"all_category_means.csv",lineterminator="\n")
    per_seed=table.groupby("seed")[METRICS].mean()
    per_seed.to_csv(output/"macro_by_seed.csv",lineterminator="\n")
    table.groupby("kind")[METRICS].mean().to_csv(output/"texture_object.csv",lineterminator="\n")
    pd.DataFrame(defects).drop(columns=["confusion_matrix"]).to_csv(output/"per_defect_results.csv",index=False,lineterminator="\n")
    uncertainty={}
    for key in METRICS:
        values=np.asarray(bootstrap[key])
        valid=np.isfinite(values).all(axis=(0,1))
        means=values[:,:,valid].mean(axis=(0,1))
        uncertainty[key]={"mean":float(per_seed[key].mean()),"seed_std":float(per_seed[key].std(ddof=1)),
                          "image_bootstrap_95ci":np.quantile(means,[.025,.975]).tolist(),
                          "bootstrap_target":"4096-bin pixel approximation" if key in quantization_bias else "exact image metric",
                          "valid_macro_replicates":int(valid.sum())}
    # Retain an identity for every local artifact actually used in aggregation.
    (output/"artifact_manifest.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    summary={"categories":15,"seeds":freeze["seeds"],"models":45,"development_runs":20,
             "frozen_configuration_sha256":sha256(ROOT/"configs/basic_cnn_frozen.json"),
             "macro":uncertainty,"bootstrap_scope":"whole-image within category; same resamples across seeds, independent categories; pixel CI 4096-bin approximation, point estimates exact",
             "pixel_bootstrap_quantization_bias":{key:{"mean_binned_minus_exact":float(np.mean(values)),
                  "maximum_absolute_run_difference":float(np.max(np.abs(values)))} for key,values in quantization_bias.items()},
             "mean_model_p50_ms":float(table.model_p50_ms.mean()),"mean_model_p95_ms":float(table.model_p95_ms.mean()),
             "mean_end_to_end_p50_ms":float(table.end_to_end_p50_ms.mean()),
             "peak_training_vram_gib":float(table.train_peak_vram_gib.max()),
             "total_training_seconds":float(table.training_seconds.sum()),
             "parameters":int(table.parameters.iloc[0]),"max_checkpoint_bytes":int(table.checkpoint_bytes.max()),
             "artifact_manifest_sha256":sha256(output/"artifact_manifest.json"),
             "configuration_changed_after_test":False,"test_evaluations_per_category_seed":1}
    summary["metric_edge_cases"]={
        "image_runs_with_no_alarms":int((~table.precision_defined).sum()),
        "pixel_runs_with_no_alarms":int((~table.pixel_precision_defined).sum()),
        "precision_convention":"Immutable foundation binary_metrics uses zero_division=0; undefined cases explicitly flagged. Do not mistake zero-alert precision for a measured probability.",
        "image_precision_defined_run_mean":float(table.loc[table.precision_defined,"precision"].mean()) if table.precision_defined.any() else None,
        "pixel_precision_defined_run_mean":float(table.loc[table.pixel_precision_defined,"pixel_precision"].mean()) if table.pixel_precision_defined.any() else None,
        "f1":"Zero when no predicted positives and actual positives exist; directly 2TP/(2TP+FP+FN)",
        "ranking_metrics":"All category test sets contain both classes; no AUROC/AP/AU-PRO undefined cases"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
