"""Predeclared schedule-supply adjustment on frozen TRAIN history/PPR scores.

No models are refitted here. Programme supply comes exclusively from broadcast
schedule rows. VAL viewing is used only by the exact existing evaluation helper,
never by supply, user profiles, graph weights or score generation. This script
has no Kaggle/network/publication action. The output folder is explicit and all
project inputs are opened read-only; the caller decides which files to retain.

Before execution, the grid was sent to the root agent and approved: two frozen
base families x gamma {0,.5,1}; smoothing=4 TRAIN-period pseudo-airings; ratio
clipped to [.25,4]. The two gamma=0 rows must reproduce existing rankings exactly.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

PREDECLARED_PLAN = {
    "families": ["history", "ppr"], "gamma": [0.0, 0.5, 1.0],
    "smoothing_train_airings": 4.0, "ratio_clip": [0.25, 4.0],
    "new_settings": 4, "gamma_zero_controls": 2,
    "expected_val_candidates": 3957, "expected_test_candidates": 6632,
    "expected_evaluation_cohort": 1216,
    "formula": "score * clip((target_count/target_days+s/train_days)/(train_count/train_days+s/train_days),.25,4)**gamma",
    "selection": "same frozen temporal VAL MAP@5; never Kaggle score or TEST viewing",
    "fitting": "none: read existing frozen TRAIN history/PPR score matrices",
    "repeats": "allowed exactly as in main experiment",
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def save(path: Path, value):
    path.write_text(json.dumps(value, default=json_default, ensure_ascii=False, indent=2) + "\n")


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def supply_multiplier(train_counts, target_counts, train_days, target_days,
                      gamma, smoothing=4.0, clip=(0.25, 4.0)):
    """Item-wise smoothed target/TRAIN airing-rate ratio, no viewing labels."""
    train_counts = np.asarray(train_counts, dtype=np.float64)
    target_counts = np.asarray(target_counts, dtype=np.float64)
    if train_counts.shape != target_counts.shape or train_counts.ndim != 1:
        raise ValueError("Supply vectors must be aligned one-dimensional arrays")
    if not np.isfinite(train_counts).all() or not np.isfinite(target_counts).all() or \
       (train_counts < 0).any() or (target_counts < 0).any():
        raise ValueError("Supply counts must be finite and nonnegative")
    if min(train_days, target_days, smoothing) <= 0 or gamma < 0:
        raise ValueError("Invalid time, smoothing or exponent")
    prior_rate = smoothing / train_days
    ratio = (target_counts / target_days + prior_rate) / (train_counts / train_days + prior_rate)
    ratio = np.clip(ratio, *clip)
    return np.power(ratio, gamma)


def run_synthetic_guards():
    # Same broadcast rate across unequal window lengths must not change a score.
    np.testing.assert_allclose(supply_multiplier([40], [10], 112, 28, 1), [1])
    np.testing.assert_allclose(supply_multiplier([40], [30], 112, 84, 1), [1])
    np.testing.assert_allclose(supply_multiplier([40], [20], 112, 28, 1), [84/44])
    np.testing.assert_allclose(supply_multiplier([0], [1], 112, 28, 1), [2])
    np.testing.assert_array_equal(supply_multiplier([0, 40], [1000, 0], 112, 28, 0), [1, 1])
    np.testing.assert_array_equal(supply_multiplier([0, 400], [1000, 0], 112, 28, 1), [4, .25])
    return {"passed": 6, "failed": 0, "scope": "AI-run synthetic formula guards"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Save the exact previously fixed grid before inspecting any extension result.
    save(output / "predeclared_supply_plan.json", {**PREDECLARED_PLAN,
        "run_started_utc": datetime.now(timezone.utc).isoformat()})
    synthetic = run_synthetic_guards()
    sys.path.insert(0, str(root / "src"))
    from sweettv.experiment import rank_rows
    from sweettv.metrics import evaluate_rankings, validate_ranked_lists, validate_submission_csv

    indices = json.loads((root / "artifacts/indices.json").read_text())
    metadata = json.loads((root / "data/processed/metadata.json").read_text())
    experiments = json.loads((root / "evidence/experiments.json").read_text())
    cohort_protocol = json.loads((root / "evidence/evaluation_protocol.json").read_text())
    interaction_protocol = json.loads((root / "evidence/interaction_profile.json").read_text())["protocol"]
    candidate_file = json.loads((root / "data/processed/candidates.json").read_text())
    item_ids = np.asarray(indices["item_ids"], dtype=np.int64)
    if [int(m["item_id"]) for m in metadata] != item_ids.tolist():
        raise ValueError("Metadata and cached score indices differ")
    imap = {int(item): i for i, item in enumerate(item_ids)}
    candidates = {split: np.asarray(candidate_file[split], dtype=np.int64) for split in ("val", "test")}
    if len(candidates["val"]) != PREDECLARED_PLAN["expected_val_candidates"] or \
       len(candidates["test"]) != PREDECLARED_PLAN["expected_test_candidates"]:
        raise ValueError("Updated starts-in-window candidate fix is not present")
    if indices["validation_candidates"] != candidates["val"].tolist() or indices["test_candidates"] != candidates["test"].tolist():
        raise ValueError("Candidate file changed after the frozen score-cache run")
    positions = {split: np.asarray([imap[int(item)] for item in ids]) for split, ids in candidates.items()}
    users = indices["target_user_ids"]
    umap = {u: i for i, u in enumerate(indices["user_ids"])}
    ui = np.asarray([umap[u] for u in users])
    X = sparse.load_npz(root / "artifacts/train_matrix.npz")
    cold = np.diff(X.indptr)[ui] == 0
    popularity = np.asarray(X.getnnz(axis=0)).ravel().astype(np.float32)
    labels = pd.read_parquet(root / "artifacts/validation_labels.parquet")
    labels["user_id"] = labels["user_id"].astype(str)
    truth = {str(u): [int(v) for v in g.item_id] for u, g in labels.groupby("user_id", sort=False)}
    frozen_cohort = [u for u in users if truth.get(u)]
    if len(frozen_cohort) != PREDECLARED_PLAN["expected_evaluation_cohort"]:
        raise ValueError("Common evaluation cohort changed")
    if cohort_protocol["final_models_refit_on_validation"]:
        raise ValueError("This extension requires the declared frozen TRAIN-only protocol")

    dates = {"train": (interaction_protocol["train_start"], interaction_protocol["validation_start"]),
             "val": (interaction_protocol["validation_start"], interaction_protocol["validation_end"]),
             "test": (interaction_protocol["hidden_test_start"], interaction_protocol["hidden_test_end"])}
    days = {split: (pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/86400 for split, (start,end) in dates.items()}
    # Database read-only is essential: no model/schedule table is edited here.
    connection = duckdb.connect(str(root / "data/processed/data.duckdb"), read_only=True)
    try:
        train_rows = connection.execute("SELECT item_id,count(DISTINCT airing_id) FROM window_airings WHERE split='train' GROUP BY item_id").fetchall()
        train_supply = np.zeros(len(item_ids), dtype=np.float64)
        for item, count in train_rows:
            train_supply[imap[int(item)]] = count
        target_supply = {}
        for split in ("val", "test"):
            source = "source_part<2" if split == "val" else "source_part=2"
            rows = connection.execute(f"SELECT item_id,count(DISTINCT airing_id) FROM airings WHERE {source} AND start_at>=? AND start_at<? GROUP BY item_id", list(dates[split])).fetchall()
            counts = np.zeros(len(item_ids), dtype=np.float64)
            for item, count in rows:
                counts[imap[int(item)]] = count
            supplied_ids = item_ids[counts > 0].tolist()
            if supplied_ids != candidates[split].tolist():
                raise ValueError(f"{split} schedule supply does not match the exact common candidate pool")
            target_supply[split] = counts
    finally:
        connection.close()

    source_hashes = {str(path.relative_to(root)): file_hash(path) for path in [
        root / "artifacts/train_matrix.npz", root / "artifacts/indices.json",
        root / "data/processed/candidates.json", root / "evidence/experiments.json"]}
    supply_provenance = {"train_source": "window_airings WHERE split='train'; DISTINCT airing_id",
        "target_source": "airings; starts-in-window; VAL source_part<2, TEST source_part=2; no views",
        "window_days": days, "train_airings": int(train_supply.sum()),
        "target_airings": {s: int(a.sum()) for s,a in target_supply.items()},
        "supply_vector_sha256": {s: hashlib.sha256(a.tobytes()).hexdigest() for s,a in {"train": train_supply, **target_supply}.items()},
        "source_hashes": source_hashes}
    save(output / "supply_provenance.json", supply_provenance)

    records, all_predictions = [], {}
    baseline_guards = []
    for family in PREDECLARED_PLAN["families"]:
        base_record = experiments["family_best"][family]
        base_key = base_record["key"]
        cache_path = root / f"artifacts/best_{family}_scores.npy"
        cache = np.load(cache_path, mmap_mode="r")
        if cache.shape != (len(users), len(item_ids)) or not np.isfinite(cache).all():
            raise ValueError("Invalid cached base score matrix")
        base_predictions = json.loads((root / f"artifacts/{base_key}_predictions.json").read_text())
        for gamma in PREDECLARED_PLAN["gamma"]:
            start = time.perf_counter()
            key = f"{family}_supply_g{str(gamma).replace('.', '_')}"
            predictions, fills = {}, {}
            split_ratios = {}
            for split, ids in candidates.items():
                multiplier = supply_multiplier(train_supply, target_supply[split], days["train"], days[split], gamma,
                    smoothing=PREDECLARED_PLAN["smoothing_train_airings"], clip=PREDECLARED_PLAN["ratio_clip"])
                scores = (np.asarray(cache[:, positions[split]], dtype=np.float64) * multiplier[positions[split]]).astype(np.float32)
                lists, fills[split] = rank_rows(scores, ids, popularity[positions[split]], cold)
                predictions[split] = dict(zip(users, lists))
                validate_ranked_lists(users, predictions[split], ids)
                if gamma == 0:
                    if predictions[split] != base_predictions[split]:
                        raise ValueError(f"Gamma-zero {family}/{split} did not reproduce frozen baseline lists")
                    baseline_guards.append({"family":family,"split":split,"exact_rankings_equal":True})
                split_ratios[split] = {"min_multiplier":float(multiplier[positions[split]].min()),
                    "median_multiplier":float(np.median(multiplier[positions[split]])),
                    "max_multiplier":float(multiplier[positions[split]].max())}
            report = evaluate_rankings(truth, predictions["val"], users, k=5)
            if report["evaluated_user_ids"] != frozen_cohort:
                raise ValueError("Comparison changed the common VAL cohort")
            if gamma == 0 and abs(report["map"] - base_record["map_at_5"]) > 1e-12:
                raise ValueError("Gamma-zero metric differs from recorded primary baseline")
            record = {"key":key,"family":family,"base_key":base_key,"base_config":base_record.get("config"),
                "gamma":gamma,"smoothing_train_airings":4.0,"ratio_clip":[.25,4.0],
                "map_at_5":report["map"],"precision_at_5":report["precision"],"recall_at_5":report["recall"],
                "evaluated_users":report["evaluated_users"],"excluded_no_relevant":report["excluded_no_relevant_count"],
                "delta_from_base_map":report["map"]-base_record["map_at_5"],"fallback":fills,
                "multiplier_summary":split_ratios,"seconds_adjust_and_rank":time.perf_counter()-start,
                "model_refit":False}
            predictions_path = output / f"{key}_predictions.json"
            save(predictions_path, predictions)
            save(output / f"{key}_validation.json", report)
            csv_path = output / f"{key}.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle);writer.writerow(["user_id","tv_show_id"])
                writer.writerows((u," ".join(map(str,predictions["test"][u]))) for u in users)
            record["submission_file"] = csv_path.name
            record["csv_validation"] = validate_submission_csv(csv_path, root / "data/raw/submission.csv", candidates["test"])
            records.append(record)
            all_predictions[key] = predictions
            print(json.dumps({k:record[k] for k in ("key","map_at_5","delta_from_base_map","seconds_adjust_and_rank")}), flush=True)
    selected = max(records, key=lambda row: row["map_at_5"])
    save(output / "supply_extension_results.json", {"predeclared_plan":PREDECLARED_PLAN,
        "records":records,"validation_selected":selected,"gamma_zero_guards":baseline_guards,
        "synthetic_guards":synthetic,"provenance":supply_provenance,
        "evaluated_user_ids_sha256":hashlib.sha256("\n".join(frozen_cohort).encode()).hexdigest(),
        "scope":"AI-run validation extension only; no TEST viewing, Kaggle score, model refit or online effect"})


if __name__ == "__main__":
    main()
