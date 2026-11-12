"""Select only the predeclared synthetic/normal development objectives and freeze."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from anomaly_detection.cnn_research import ROOT, code_identity
from anomaly_detection.data import CATEGORIES, sha256
from anomaly_detection.tracking import config_hash, record


def main():
    plan=json.loads((ROOT/"configs/basic_cnn_plan.json").read_text())
    rows=[]; runs=[]
    for variant in plan["variants"]:
        for category in plan["development_categories"]:
            identifier=("01-pilot-base-bottle-s2026-v1" if variant["name"]=="base" and category=="bottle"
                        else f"01-dev-{variant['name']}-{category}-s2026-v1")
            result_path=ROOT/f"reports/basic_cnn/runs/{identifier}-result.json"
            result=json.loads(result_path.read_text())
            assert result["configuration_sha256"]==config_hash(result["configuration"])
            assert sha256(ROOT/result["checkpoint"])==result["checkpoint_sha256"]
            assert sha256(ROOT/result["history"])==result["history_sha256"]
            metrics=result["selected_development_metrics"]
            rows.append({"variant":variant["name"],"category":category,"experiment_id":identifier,
                         "proxy_recall":metrics["proxy_recall_at_dev_normal_q95"],
                         "proxy_pixel_ap":metrics["proxy_pixel_ap"],"proxy_image_auroc":metrics["proxy_image_auroc"],
                         "best_epoch":result["best_epoch"],"parameters":result["parameters"],
                         "peak_vram_gib":result["peak_allocated_bytes"]/2**30,
                         "training_seconds":result["training_seconds_including_loading_validation"]})
            runs.append(result)
    table=pd.DataFrame(rows)
    table.to_csv(ROOT/"reports/basic_cnn/development_comparison.csv",index=False,lineterminator="\n")
    comparison=table.groupby("variant").agg(proxy_recall=("proxy_recall","mean"),
        proxy_pixel_ap=("proxy_pixel_ap","mean"),parameters=("parameters","first"),
        best_epoch=("best_epoch","median"),peak_vram_gib=("peak_vram_gib","max"),training_seconds=("training_seconds","sum"))
    comparison=comparison.sort_values(["proxy_recall","proxy_pixel_ap","parameters"],ascending=[False,False,True])
    comparison.to_csv(ROOT/"reports/basic_cnn/development_macro.csv",lineterminator="\n")
    chosen=comparison.index[0]
    base=next(r["configuration"] for r in runs if r["configuration"]["name"]==chosen)
    epochs=int(np.clip(round(comparison.loc[chosen,"best_epoch"]),3,8))
    config=dict(base,max_epochs=epochs)
    freeze={"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"selected_variant":chosen,
            "configuration":config,"configuration_sha256":config_hash(config),"seeds":plan["final_seeds"],
            "categories":list(CATEGORIES),"planned_fresh_models":len(CATEGORIES)*len(plan["final_seeds"]),
            "final_stopping":"fixed epoch count, selected from predeclared development-category median; no test/category-specific changes",
            "selection":"macro synthetic-probe recall at official-normal development q95; AP/parameter tie-breaks",
            "development_run_ids":table.experiment_id.tolist(),
            "development_table_sha256":sha256(ROOT/"reports/basic_cnn/development_comparison.csv"),
            "plan_sha256":sha256(ROOT/"configs/basic_cnn_plan.json"),"code":code_identity(),
            "evaluation_source_sha256":sha256(ROOT/"src/anomaly_detection/cnn_evaluation.py"),
            "pixel_metric_source_sha256":sha256(ROOT/"src/anomaly_detection/pixel_metrics.py"),
            "image_threshold_policy":"normal calibration q95 higher; secondary q99, both strict >",
            "pixel_threshold_policy":"normal calibration pooled q99.5, conservative 65536-bin upper edge",
            "test_access_during_selection":False,"resolution_experiment":"not conducted; pilot learned, no extra search justified"}
    target=ROOT/"configs/basic_cnn_frozen.json"
    with target.open("x") as stream:json.dump(freeze,stream,indent=2,allow_nan=False)
    (ROOT/"reports/basic_cnn/freeze_identity.json").write_text(json.dumps({"sha256":sha256(target)},indent=2)+"\n")
    for result in runs:
        identifier=result["experiment_id"]
        event=json.loads((ROOT/f"experiments/{identifier}-result.json").read_text())
        event.pop("recorded_at");event.pop("config_sha256")
        selected=result["configuration"]["name"]==chosen
        event.update(experiment_id=identifier+"-decision",status="reviewed",
                     decision="keep" if selected else "reject",
                     interpretation=f"Predeclared macro proxy ranking selected {chosen}; this experiment is retained regardless of decision.")
        record(ROOT/"experiments",event)
    print(comparison.to_string())
    print(json.dumps(freeze,indent=2))


if __name__=="__main__":main()
