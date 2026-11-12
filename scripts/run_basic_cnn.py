"""Explicit phase driver; never silently advances from pilot to all-category fits."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomaly_detection.cnn_research import ROOT, train_experiment
from anomaly_detection.data import sha256
from anomaly_detection.cnn_evaluation import evaluate_frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["pilot", "development", "final-train", "final-evaluate"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "configs/basic_cnn_plan.json").read_text())
    base = {k: plan[k] for k in ("seed", "resolution", "batch_size", "learning_rate", "weight_decay",
            "max_epochs", "minimum_epochs", "patience", "normal_probability", "positive_weight")}
    if args.phase == "pilot":
        base.update(plan["variants"][0])
        result = train_experiment("01-pilot-base-bottle-s2026-v1", "bottle", base,
            "A scratch CNN can localize held-out fitting-derived synthetic edits while suppressing real normal development images.", phase="pilot")
        print(json.dumps(result, indent=2))
    elif args.phase == "development":
        started = time.perf_counter()
        for variant in plan["variants"]:
            for category in plan["development_categories"]:
                # Explicit pilot reuse, not a new unlogged rerun.
                if variant["name"] == "base" and category == "bottle":
                    continue
                config = dict(base, **variant)
                identifier = f"01-dev-{variant['name']}-{category}-s2026-v1"
                if (ROOT / f"experiments/{identifier}-result.json").exists():
                    print("Existing immutable completed experiment:", identifier, flush=True)
                    continue
                train_experiment(identifier, category, config,
                    f"Controlled {variant['name']} ablation; compare only fixed fitting-derived proxy and normal-development objectives.")
                if time.perf_counter()-started > 60*60:
                    raise RuntimeError("Development wall budget exceeded; do not expand silently")
    else:
        frozen_path=ROOT/"configs/basic_cnn_frozen.json"
        frozen=json.loads(frozen_path.read_text())
        freeze_hash=sha256(frozen_path)
        expected=json.loads((ROOT/"reports/basic_cnn/freeze_identity.json").read_text())["sha256"]
        assert freeze_hash==expected
        training_wall=0.
        for seed in frozen["seeds"]:
            for category in frozen["categories"]:
                variant=frozen["selected_variant"]
                identifier=f"01-final-{variant}-{category}-s{seed}-v1"
                path=ROOT/f"reports/basic_cnn/runs/{identifier}-result.json"
                if args.phase == "final-train" and not path.exists():
                    config=dict(frozen["configuration"],seed=seed)
                    result=train_experiment(identifier,category,config,
                        "Fresh per-category/per-seed confirmation under frozen policy; no calibration/test selection.",phase="final")
                    training_wall+=result["training_seconds_including_loading_validation"]
                    if training_wall>60*60:raise RuntimeError("Final training wall budget exceeded")
                else:
                    result=json.loads(path.read_text())
                    assert sha256(ROOT/result["checkpoint"])==result["checkpoint_sha256"]
                if args.phase=="final-evaluate":
                    eval_id=f"01-test-{variant}-{category}-s{seed}-v1"
                    completed=ROOT/f"reports/basic_cnn/evaluation/{eval_id}-result.json"
                    if completed.exists():
                        print("Existing immutable evaluation; not repeated:",eval_id,flush=True)
                        continue
                    metrics=evaluate_frozen(eval_id,result,freeze_hash)
                    print(json.dumps({k:metrics[k] for k in ("experiment_id","category","seed","image_auroc","image_ap","pixel_auroc","pixel_ap","au_pro","recall","fpr","evaluation_seconds")}),flush=True)


if __name__ == "__main__":
    main()
