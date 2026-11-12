"""Auditable Basic CNN training, partition isolation and immutable artifacts."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import random
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .basic_cnn import BasicCNN, seed_everything, synthetic_batch, segmentation_loss, image_scores
from .data import CATEGORIES, autodiscover_data_root, sha256
from .metrics import binary_metrics, calibrate_normal
from .tracking import config_hash, record

ROOT = Path(__file__).resolve().parents[2]


def read_partition(partition, category, project=ROOT):
    if partition not in {"fit", "development", "calibration", "test"} or category not in CATEGORIES:
        raise ValueError("Unknown partition/category")
    metadata = json.loads((project / "reports/dataset_validation.json").read_text())
    spec = metadata["partitions"][partition]
    path = project / spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError("Frozen manifest hash mismatch")
    rows = pd.read_csv(path)
    rows = rows[rows.category.eq(category)].sort_values("path").reset_index(drop=True)
    if rows.empty or not rows.partition.eq(partition).all():
        raise ValueError("Empty or inconsistent category partition")
    if partition != "test":
        if not rows.split.eq("train").all() or not rows.path.str.contains("/train/good/").all() or rows["mask"].notna().any():
            raise ValueError("Non-normal file in normal partition")
    return rows


def inner_fit_split(rows, seed=2026, fraction=.2):
    require_fitting(rows)
    positions = list(range(len(rows)))
    random.Random(f"inner:{seed}:{rows.category.iloc[0]}").shuffle(positions)
    count = max(1, int(len(rows)*fraction))
    return (rows.iloc[positions[count:]].sort_values("path").reset_index(drop=True),
            rows.iloc[positions[:count]].sort_values("path").reset_index(drop=True))


def require_fitting(rows):
    if rows.empty or not rows.partition.eq("fit").all() or not rows.split.eq("train").all() or rows["mask"].notna().any():
        raise ValueError("Synthetic/training sources must be fitting normals only")


def load_resized(rows, resolution, allow_test=False):
    """Fixed PIL bilinear antialiased resize; cached uint8, then float /255.

    Cache identity includes every original path/hash and preprocessing version.
    Original SHA256 is checked when a cache is first built. No labels are read.
    """
    if not allow_test and not rows.split.eq("train").all():
        raise ValueError("Test images forbidden in training/development loading")
    dataset_root = autodiscover_data_root(ROOT)
    key = config_hash({"records": rows[["path", "sha256"]].to_dict("records"),
                       "resolution": resolution, "preprocessing": "PIL-RGB-bilinear-v1"})
    folder = ROOT / "outputs/basic_cnn/cache"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{key}.npy"
    if not target.exists():
        images = []
        for row in rows.itertuples():
            path = dataset_root / row.path
            if not path.resolve().is_relative_to(dataset_root) or sha256(path) != row.sha256:
                raise ValueError("Source file escaped root or changed since validation")
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
                images.append(np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1).copy())
        np.save(target, np.stack(images), allow_pickle=False)
    return torch.from_numpy(np.load(target, allow_pickle=False))


def make_probes(images, rows, seed=9901):
    require_fitting(rows)
    generator = torch.Generator(device=images.device).manual_seed(seed)
    normal = images.float()/255
    changed, masks = synthetic_batch(normal, generator, "mixed", normal_probability=0.)
    return torch.cat((normal, changed)), torch.cat((torch.zeros_like(masks), masks))


def train_epoch(model, images, optimizer, generator, shuffle_generator, config):
    model.train()
    permutation = torch.randperm(len(images), generator=shuffle_generator)
    loss_sum, grad_sum = 0., 0.
    for start in range(0, len(images), config["batch_size"]):
        indices = permutation[start:start+config["batch_size"]].to(images.device)
        normal = images[indices].float()/255
        inputs, masks = synthetic_batch(normal, generator, config["synthetic"], config["normal_probability"])
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = segmentation_loss(logits, masks, config["positive_weight"], config["dice_weight"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += float(loss.detach())*len(inputs)
        grad_sum += float(norm.detach())*len(inputs)
    return {"train_loss": loss_sum/len(images), "gradient_norm": grad_sum/len(images)}


@torch.inference_mode()
def predict_logits(model, images, batch_size=16, uint8=True):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size]
        predictions.append(model(batch.float()/255 if uint8 else batch).float().cpu())
    return torch.cat(predictions)


def development_metrics(model, normals, probes, probe_masks, config):
    normal_logits = predict_logits(model, normals, config["batch_size"])
    proxy_logits = predict_logits(model, probes, config["batch_size"], uint8=False)
    normal_loss = float(F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits)))
    proxy_loss = float(segmentation_loss(proxy_logits, probe_masks.cpu(), config["positive_weight"], config["dice_weight"]))
    normal_scores = image_scores(normal_logits).numpy()
    proxy_scores = image_scores(proxy_logits).numpy()
    labels = probe_masks.cpu().flatten(1).any(1).long().numpy()
    threshold = calibrate_normal(normal_scores)
    metrics = binary_metrics(labels, proxy_scores, threshold)
    pixel_ap = average_precision_score(probe_masks.cpu().numpy().ravel(), proxy_logits.sigmoid().numpy().ravel())
    return {"normal_development_bce": normal_loss, "probe_loss": proxy_loss,
            "stopping_objective": .5*(normal_loss+proxy_loss),
            "proxy_image_auroc": metrics["auroc"], "proxy_image_ap": metrics["ap"],
            "proxy_recall_at_dev_normal_q95": metrics["recall"], "proxy_pixel_ap": float(pixel_ap),
            "development_threshold": threshold, "probe_positives": int(labels.sum()),
            "normal_development_images": len(normals)}


def code_identity():
    files = [Path(__file__), Path(__file__).with_name("basic_cnn.py")]
    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "source_sha256": {f.relative_to(ROOT).as_posix(): sha256(f) for f in files},
            "environment_lock_sha256": sha256(ROOT / "requirements.lock.txt")}


def train_experiment(identifier, category, config, hypothesis, phase="development"):
    """One immutable start/result/failure pair; no calibration/test access here."""
    if phase not in {"pilot", "development", "final"}:
        raise ValueError("Unknown phase")
    directory = ROOT / "outputs/basic_cnn" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = ROOT / "checkpoints/basic_cnn" / identifier
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    compact = ROOT / "reports/basic_cnn/runs"
    compact.mkdir(parents=True, exist_ok=True)
    environment = json.loads((ROOT / "reports/environment.json").read_text())
    identity = code_identity()
    metadata = json.loads((ROOT / "reports/dataset_validation.json").read_text())
    entry = {"experiment_id": identifier+"-start", "model_family": "BasicCNN",
             "categories": [category], "hypothesis": hypothesis,
             "preprocessing": {"RGB": True, "resize": config["resolution"], "float_range": [0,1],
                               "model_centering": "2*x-1", "resize_method": "PIL bilinear antialiased"},
             "hyperparameters": config, "seed": config["seed"], "environment": environment,
             "training": {"phase": phase, "code": identity, "partition_identities": metadata["partitions"]},
             "metrics": {}, "calibration": {"used": False},
             "paths": {"output": directory.relative_to(ROOT).as_posix(),
                       "checkpoints": checkpoint_dir.relative_to(ROOT).as_posix()},
             "interpretation": "Reserved before training; synthetic proxies are not benchmark results",
             "decision": "pending", "status": "started"}
    record(ROOT / "experiments", entry)
    (directory / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    for name in identity["source_sha256"]:
        (directory / Path(name).name).write_bytes((ROOT / name).read_bytes())
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no silent CPU fallback")
        seed_everything(config["seed"])
        fitting = read_partition("fit", category)
        development = read_partition("development", category)
        if phase == "final":
            learning_rows = fitting
            probe_rows = None
        else:
            learning_rows, probe_rows = inner_fit_split(fitting)
        learning_rows.to_csv(directory / "gradient_sources.csv", index=False, lineterminator="\n")
        learning = load_resized(learning_rows, config["resolution"]).cuda()
        normals = load_resized(development, config["resolution"]).cuda()
        if probe_rows is not None:
            probe_rows.to_csv(directory / "synthetic_probe_sources.csv", index=False, lineterminator="\n")
            probe_uint8 = load_resized(probe_rows, config["resolution"]).cuda()
            probes, masks = make_probes(probe_uint8, probe_rows)
        model = BasicCNN(config["width"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generator = torch.Generator(device="cuda").manual_seed(config["seed"]+101)
        shuffle = torch.Generator().manual_seed(config["seed"]+102)
        torch.cuda.reset_peak_memory_stats()
        history, best, stale = [], float("inf"), 0
        best_epoch = 0
        for epoch in range(1, config["max_epochs"]+1):
            epoch_start = time.perf_counter()
            metrics = train_epoch(model, learning, optimizer, generator, shuffle, config)
            torch.cuda.synchronize()
            metrics["train_epoch_seconds"] = time.perf_counter()-epoch_start
            if phase != "final":
                metrics.update(development_metrics(model, normals, probes, masks, config))
            else:
                normal_logits = predict_logits(model, normals, config["batch_size"])
                metrics["normal_development_bce"] = float(F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits)))
            metrics.update(epoch=epoch, elapsed_seconds=time.perf_counter()-started,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            history.append(metrics)
            pd.DataFrame(history).to_csv(compact / f"{identifier}-history.csv", index=False, lineterminator="\n")
            print(json.dumps(dict(experiment=identifier, **metrics)), flush=True)
            snapshot = {"state_dict": model.state_dict(), "config": config, "category": category,
                        "epoch": epoch, "training_phase": phase, "code_identity": identity}
            if phase == "final" or metrics["stopping_objective"] < best-1e-4:
                best = metrics.get("stopping_objective", 0.)
                best_epoch = epoch
                stale = 0
                torch.save(snapshot, checkpoint_dir / "best.pt")
            else:
                stale += 1
            if metrics["peak_allocated_bytes"] > 6*2**30:
                raise RuntimeError("Preregistered 6 GiB memory budget exceeded")
            if time.perf_counter()-started > 20*60:
                raise RuntimeError("Per-run 20-minute wall budget exceeded")
            if phase != "final" and epoch >= config["minimum_epochs"] and stale >= config["patience"]:
                break
        torch.save(snapshot, checkpoint_dir / "last.pt")
        # Optimizer/RNG state allows diagnosing or explicitly registered resumption.
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "synthetic_rng": generator.get_state(),
                    "shuffle_rng": shuffle.get_state()}, checkpoint_dir / "last-training-state.pt")
        chosen = copy.deepcopy(history[best_epoch-1])
        summary = {"experiment_id": identifier, "category": category, "phase": phase,
                   "configuration": config, "configuration_sha256": config_hash(config),
                   "best_epoch": best_epoch, "epochs_completed": len(history),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "gradient_normal_images": len(learning_rows),
                   "probe_normal_sources": len(probe_rows) if probe_rows is not None else 0,
                   "training_seconds_including_loading_validation": time.perf_counter()-started,
                   "training_gpu_seconds": sum(h["train_epoch_seconds"] for h in history),
                   "selected_development_metrics": chosen,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "checkpoint": (checkpoint_dir / "best.pt").relative_to(ROOT).as_posix(),
                   "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"),
                   "checkpoint_bytes": (checkpoint_dir / "best.pt").stat().st_size,
                   "history": (compact / f"{identifier}-history.csv").relative_to(ROOT).as_posix(),
                   "history_sha256": sha256(compact / f"{identifier}-history.csv"),
                   "code_identity": identity}
        with (compact / f"{identifier}-result.json").open("x") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        result_entry = copy.deepcopy(entry)
        result_entry.update(experiment_id=identifier+"-result", status="completed",
                            metrics=summary, decision="await bounded comparison" if phase != "final" else "frozen confirmation trained",
                            interpretation="Synthetic development results only" if phase != "final" else "Fresh category-specific model; test not accessed")
        record(ROOT / "experiments", result_entry)
        return summary
    except BaseException as exc:
        details = traceback.format_exc()
        (directory / "failure.txt").write_text(details, encoding="utf-8")
        failed = copy.deepcopy(entry)
        failed.update(experiment_id=identifier+"-failure", status="failed", decision="investigate; no silent config changes",
                      interpretation=f"{type(exc).__name__}: {exc}")
        record(ROOT / "experiments", failed)
        if isinstance(exc, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        raise
