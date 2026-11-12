"""Validate completed research artifacts without repeating official test inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import re
import sys
import inspect
import numpy as np
import pandas as pd
import nbformat
import torch
from scipy.special import digamma
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from anomaly_detection.data import CATEGORIES,sha256
from anomaly_detection.tracking import config_hash
from anomaly_detection.metrics import binary_metrics,calibrate_normal
from anomaly_detection.cnn_research import ROOT,read_partition,require_fitting
from anomaly_detection.basic_cnn import BasicCNN,seed_everything
from anomaly_detection.cnn_evaluation import load_checkpoint,calibrate_model


def validate_notebook(require_executed=True):
    path=ROOT/"notebooks/01_basic_cnn.ipynb"
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    headings=[int(n) for cell in notebook.cells if cell.cell_type=="markdown"
              for n in re.findall(r"^## (\d+)\.",cell.source,re.M)]
    assert headings==list(range(1,28))
    code=0;executed=0;private=0
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type!="code":continue
        code+=1;compile(cell.source,f"01_basic_cnn.ipynb:{i}","exec")
        assert notebook.cells[i-1].cell_type==notebook.cells[i+1].cell_type=="markdown"
        assert len(notebook.cells[i-1].source)>80 and len(notebook.cells[i+1].source)>80
        assert not any(o.output_type=="error" for o in cell.outputs)
        if cell.execution_count is not None:executed+=1
        if cell.metadata.get("contains_dataset_images"):
            private+=1
            if require_executed:assert not cell.outputs,"Licensed image panels stay local"
    assert private==2
    assert any(cell.cell_type=="code" and cell.source.startswith(inspect.getsource(BasicCNN).strip()) for cell in notebook.cells)
    assert any(cell.cell_type=="code" and "def train_epoch(" in cell.source and "def train_experiment(" in cell.source for cell in notebook.cells)
    if require_executed:assert code==executed==27
    assert path.stat().st_size<3_000_000
    return {"sections":27,"code_cells":code,"executed_code_cells":executed,"bytes":path.stat().st_size}


def validate_artifacts(verify_calibration=False):
    frozen_path=ROOT/"configs/basic_cnn_frozen.json"
    frozen=json.loads(frozen_path.read_text());freeze_digest=sha256(frozen_path)
    assert freeze_digest==json.loads((ROOT/"reports/basic_cnn/freeze_identity.json").read_text())["sha256"]
    assert frozen["categories"]==list(CATEGORIES) and frozen["seeds"]==[2026,2027,2028]
    assert sha256(ROOT/"src/anomaly_detection/cnn_evaluation.py")==frozen["evaluation_source_sha256"]
    assert sha256(ROOT/"src/anomaly_detection/pixel_metrics.py")==frozen["pixel_metric_source_sha256"]
    for path,digest in frozen["code"]["source_sha256"].items():assert sha256(ROOT/path)==digest
    registry=list((ROOT/"experiments").glob("01-*.json"))
    for path in registry:
        item=json.loads(path.read_text());digest=item.pop("config_sha256");item.pop("recorded_at")
        assert config_hash(item)==digest,path.name
    original_ids=list(frozen["development_run_ids"])+[
        f"01-final-{frozen['selected_variant']}-{category}-s{seed}-v1"
        for seed in frozen["seeds"] for category in CATEGORIES]
    run_paths=[ROOT/f"reports/basic_cnn/runs/{identifier}-result.json" for identifier in original_ids]
    assert len(run_paths)==65
    final_count=0
    for path in run_paths:
        run=json.loads(path.read_text());identifier=run["experiment_id"]
        assert config_hash(run["configuration"])==run["configuration_sha256"]
        assert sha256(ROOT/run["checkpoint"])==run["checkpoint_sha256"]
        assert sha256(ROOT/run["history"])==run["history_sha256"]
        local=ROOT/"outputs/basic_cnn"/identifier
        rows=pd.read_csv(local/"gradient_sources.csv");require_fitting(rows)
        assert set(rows.category)=={run["category"]}
        outer=read_partition("fit",run["category"])
        assert set(rows.sha256).issubset(set(outer.sha256))
        if run["phase"]=="final":
            final_count+=1
            assert run["configuration"]==dict(frozen["configuration"],seed=run["configuration"]["seed"])
            assert set(rows.sha256)==set(outer.sha256)
        else:
            probes=pd.read_csv(local/"synthetic_probe_sources.csv");require_fitting(probes)
            assert not set(probes.sha256)&set(rows.sha256)
            assert set(probes.sha256)|set(rows.sha256)==set(outer.sha256)
        for name,digest in run["code_identity"]["source_sha256"].items():
            assert sha256(local/Path(name).name)==digest
        checkpoint=torch.load(ROOT/run["checkpoint"],map_location="cpu",weights_only=True)
        assert checkpoint["category"]==run["category"] and checkpoint["config"]==run["configuration"]
        assert checkpoint["epoch"]==run["best_epoch"]
    assert final_count==45
    result_paths=[ROOT/f"reports/basic_cnn/evaluation/01-test-{frozen['selected_variant']}-{category}-s{seed}-v1-result.json"
                  for seed in frozen["seeds"] for category in CATEGORIES]
    assert len(result_paths)==45
    seen=set()
    for path in result_paths:
        result=json.loads(path.read_text());category=result["category"];seed=result["seed"]
        assert (category,seed) not in seen;seen.add((category,seed))
        assert result["freeze_sha256"]==freeze_digest
        identifier=result["experiment_id"]
        rows=pd.read_csv(ROOT/f"reports/basic_cnn/evaluation/{identifier}-images.csv")
        # CSV stores round-trippable float32 decimals; restore the actual score
        # dtype before strict threshold comparisons, especially on exact ties.
        rows["score"]=rows["score"].astype(np.float32)
        official=read_partition("test",category)
        assert rows.path.tolist()==official.path.tolist() and len(rows)==len(official)
        calibration=result["calibration"]
        cal_rows=read_partition("calibration",category)
        assert calibration["images"]==len(cal_rows)==len(calibration["calibration_scores"])
        assert calibration["source_paths_sha256"]==config_hash(cal_rows[["path","sha256"]].to_dict("records"))
        assert calibrate_normal(calibration["calibration_scores"])==calibration["image_threshold"]
        metrics=binary_metrics(rows.label,rows.score,calibration["image_threshold"])
        for key in ("recall","fpr","precision","f1"):assert np.isclose(metrics[key],result[key])
        assert np.isclose(metrics["auroc"],result["image_auroc"])
        assert np.isclose(metrics["ap"],result["image_ap"])
        assert metrics["confusion_matrix"]==result["confusion_matrix"]
        maps=ROOT/result["local_maps"];assert sha256(maps)==result["maps_sha256"]
        retained_maps=np.load(maps,mmap_mode="r")
        assert retained_maps.shape==(len(rows),256,256)
        np.testing.assert_array_equal(retained_maps.reshape(len(rows),-1).max(1),rows.score.to_numpy())
        local=maps.parent
        with np.load(local/"native_image_histograms.npz") as hist:
            positive=hist["positive"].sum(0,dtype=np.int64);negative=hist["negative"].sum(0,dtype=np.int64)
            assert positive.sum()==result["positive_pixels"] and negative.sum()==result["negative_pixels"]
            assert np.isclose(hist["region"].sum(),hist["region_counts"].sum())
            # Exact AUROC lies within this provable within-bin tie uncertainty.
            bound=.5*np.dot(positive.astype(float),negative.astype(float))/(positive.sum()*negative.sum())
            binned=result["bootstrap_binned_pixel_point_estimates"]["pixel_auroc"]
            assert abs(result["pixel_auroc"]-binned)<=bound+1e-9
            p=positive[::-1].astype(float);n=negative[::-1].astype(float)
            prior_p=np.cumsum(p)-p;prior_n=np.cumsum(n)-n
            def ap_ordering(neg_before):
                denominator=prior_p+neg_before
                harmonic=digamma(denominator+p+1)-digamma(denominator+1)
                return float((p-neg_before*harmonic).sum()/p.sum())
            lower=ap_ordering(prior_n+n);upper=ap_ordering(prior_n)
            assert lower-1e-6<=result["pixel_ap"]<=upper+1e-6
        times=pd.read_csv(local/"latency_ms.csv")
        assert len(times)==200 and (times>0).all().all()
        for column in times:
            assert np.isclose(np.median(times[column]),result["latency"][column]["p50_ms"])
            assert np.isclose(np.quantile(times[column],.95),result["latency"][column]["p95_ms"])
        assert result["training_peak_allocated_bytes"]<6*2**30
        start=json.loads((ROOT/f"experiments/{identifier}-start.json").read_text())
        assert start["recorded_at"]>=frozen["frozen_at_utc"]
        if verify_calibration:
            training=json.loads((ROOT/f"reports/basic_cnn/runs/{result['training_experiment']}-result.json").read_text())
            seed_everything(seed);model,_=load_checkpoint(ROOT/training["checkpoint"])
            recalibrated=calibrate_model(model,category,256,seed)
            np.testing.assert_allclose(recalibrated["calibration_scores"],calibration["calibration_scores"],atol=1e-6,rtol=1e-5)
            assert abs(recalibrated["pixel_threshold"]-calibration["pixel_threshold"])<2/65536
            del model;torch.cuda.empty_cache()
    artifacts=json.loads((ROOT/"reports/basic_cnn/artifact_manifest.json").read_text())
    for item in artifacts:
        assert sha256(ROOT/item["path"])==item["sha256"]
        assert (ROOT/item["path"]).stat().st_size==item["bytes"]
    return {"registry_events":len(registry),"training_runs":65,"fresh_final_models":45,
            "official_evaluations":45,"artifact_identities_verified":len(artifacts),
            "calibration_reproduced_from_normals":verify_calibration,"test_inference_repeated":False,
            "verified_at_utc":datetime.now(timezone.utc).isoformat()}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify-calibration",action="store_true")
    args=parser.parse_args()
    report=validate_artifacts(args.verify_calibration);report["notebook"]=validate_notebook()
    (ROOT/"reports/basic_cnn/verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
