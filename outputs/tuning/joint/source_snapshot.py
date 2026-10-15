"""Run reproducible tuning rounds after the broad comparison in main.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "project1-matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import main


PROJECT_DIR = Path(__file__).resolve().parents[1]
INITIAL_DIR = PROJECT_DIR / "outputs" / "restarted"
TUNING_DIR = PROJECT_DIR / "outputs" / "tuning"
LOGGER = logging.getLogger("project1-tuning")
REGULARIZATION_C_VALUES = [2.0, 5.0, 15.0, 30.0, 50.0, 100.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def regularization_candidates() -> list[dict]:
    vectorizer = {
        "analyzer": "word", "ngram_range": [1, 1], "max_features": 20_000,
        "min_df": 1, "sublinear_tf": True,
    }
    return [
        {
            "candidate_id": f"regularization__word_1_1_20k_sublinear__LogisticRegression__C_{c:g}",
            "vectorizer": vectorizer, "classifier": "LogisticRegression", "C": c,
        }
        for c in REGULARIZATION_C_VALUES
    ]


def feature_candidates(c_value: float) -> list[dict]:
    configs = [
        ("word_1_1_30k", {"analyzer": "word", "ngram_range": [1, 1], "max_features": 30_000, "min_df": 1}),
        ("word_1_1_50k", {"analyzer": "word", "ngram_range": [1, 1], "max_features": 50_000, "min_df": 1}),
        ("word_1_1_all", {"analyzer": "word", "ngram_range": [1, 1], "max_features": None, "min_df": 1}),
        ("word_1_1_50k_min2", {"analyzer": "word", "ngram_range": [1, 1], "max_features": 50_000, "min_df": 2}),
        ("word_1_2_30k_min2", {"analyzer": "word", "ngram_range": [1, 2], "max_features": 30_000, "min_df": 2}),
        ("word_1_2_50k_min2", {"analyzer": "word", "ngram_range": [1, 2], "max_features": 50_000, "min_df": 2}),
    ]
    candidates = []
    for config_id, vectorizer in configs:
        vectorizer = {**vectorizer, "sublinear_tf": True}
        candidates.append({
            "candidate_id": f"features__{config_id}_sublinear__LogisticRegression__C_{c_value:g}",
            "vectorizer": vectorizer, "classifier": "LogisticRegression", "C": c_value,
        })
    return candidates


def joint_candidates() -> list[dict]:
    """Refine the region around the best 50k-feature, C=30 candidate."""
    candidates = []
    for max_features in (40_000, 50_000, 60_000, 80_000):
        for c_value in (20.0, 30.0, 40.0, 60.0):
            if max_features == 50_000 and c_value == 30.0:
                continue  # Already evaluated in the feature round.
            candidates.append({
                "candidate_id": (
                    f"joint__word_1_1_{max_features // 1000}k_sublinear"
                    f"__LogisticRegression__C_{c_value:g}"
                ),
                "vectorizer": {
                    "analyzer": "word", "ngram_range": [1, 1],
                    "max_features": max_features, "min_df": 1, "sublinear_tf": True,
                },
                "classifier": "LogisticRegression", "C": c_value,
            })
    return candidates


def build_pipeline(candidate: dict) -> Pipeline:
    vectorizer = dict(candidate["vectorizer"])
    vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", **vectorizer)),
        ("classifier", LogisticRegression(
            C=float(candidate["C"]), max_iter=2_000, random_state=main.RANDOM_STATE,
        )),
    ])


def load_fixed_splits(X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    path = INITIAL_DIR / "cv_splits.csv"
    assignments = pd.read_csv(path)
    if list(assignments.columns) != ["row_index", "target", "validation_fold"]:
        raise ValueError("Unexpected cv_splits.csv structure.")
    if len(assignments) != len(X):
        raise ValueError("CV assignments do not match the training data length.")
    np.testing.assert_array_equal(assignments["row_index"].to_numpy(), np.arange(len(X)))
    np.testing.assert_array_equal(assignments["target"].to_numpy(), y)
    splits = []
    for fold in range(1, main.N_SPLITS + 1):
        valid = np.flatnonzero(assignments["validation_fold"].to_numpy() == fold)
        train = np.flatnonzero(assignments["validation_fold"].to_numpy() != fold)
        splits.append((train, valid))
    return splits


def best_regularization_c() -> float:
    initial = json.loads((INITIAL_DIR / "selected_config.json").read_text(encoding="utf-8"))
    candidates = [{
        "C": float(initial["candidate"]["parameters"]["C"]),
        "macro_f1_mean": float(initial["cv_metrics"]["macro_f1_mean"]),
        "macro_f1_std": float(initial["cv_metrics"]["macro_f1_std"]),
    }]
    regularization = pd.read_csv(TUNING_DIR / "regularization" / "cv_results.csv")
    candidates.extend(regularization[["C", "macro_f1_mean", "macro_f1_std"]].to_dict("records"))
    ordered = sorted(candidates, key=lambda row: (-row["macro_f1_mean"], row["macro_f1_std"], row["C"]))
    return float(ordered[0]["C"])


def run_candidates(
    candidates: list[dict], X: np.ndarray, y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]], output_dir: Path, jobs: int,
) -> pd.DataFrame:
    summaries, fold_rows = [], []
    for number, candidate in enumerate(candidates, start=1):
        LOGGER.info("Candidate %d/%d: %s", number, len(candidates), candidate["candidate_id"])
        started = perf_counter()
        scores = cross_validate(
            build_pipeline(candidate), X, y, cv=splits,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=jobs, return_train_score=True, error_score="raise",
        )
        row = {
            "candidate_id": candidate["candidate_id"], "C": candidate["C"],
            "vectorizer": json.dumps(candidate["vectorizer"], sort_keys=True),
            "wall_time_seconds": perf_counter() - started,
        }
        for metric in ("accuracy", "macro_f1"):
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std(ddof=0))
            row[f"train_{metric}_mean"] = float(scores[f"train_{metric}"].mean())
        summaries.append(row)
        for index in range(main.N_SPLITS):
            fold_rows.append({
                "candidate_id": candidate["candidate_id"], "fold": index + 1,
                "validation_accuracy": scores["test_accuracy"][index],
                "validation_macro_f1": scores["test_macro_f1"][index],
                "train_accuracy": scores["train_accuracy"][index],
                "train_macro_f1": scores["train_macro_f1"][index],
                "fit_time_seconds": scores["fit_time"][index],
                "score_time_seconds": scores["score_time"][index],
            })
        pd.DataFrame(summaries).to_csv(output_dir / "cv_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_results.csv", index=False)
        LOGGER.info(
            "Macro-F1 %.4f ± %.4f; Accuracy %.4f",
            row["macro_f1_mean"], row["macro_f1_std"], row["accuracy_mean"],
        )
    results = pd.DataFrame(summaries).sort_values(
        ["macro_f1_mean", "macro_f1_std", "candidate_id"], ascending=[False, True, True],
    ).reset_index(drop=True)
    results.to_csv(output_dir / "cv_results.csv", index=False)
    selected_id = results.iloc[0]["candidate_id"]
    selected = next(candidate for candidate in candidates if candidate["candidate_id"] == selected_id)
    write_json(output_dir / "selected_config.json", {
        "candidate": selected, "cv_metrics": results.iloc[0].to_dict(),
        "selection_rule": "highest macro_f1_mean, then lowest macro_f1_std, then candidate_id",
    })
    return results


def run_round(round_name: str, jobs: int) -> None:
    train_path = PROJECT_DIR / "data-Project1" / "train_data.csv"
    X, y = main.load_training_data(train_path)
    splits = load_fixed_splits(X, y)
    output_dir = TUNING_DIR / round_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if round_name == "regularization":
        candidates = regularization_candidates()
        based_on = "Initial best: 20k word unigrams, sublinear TF, C=20"
    elif round_name == "features":
        c_value = best_regularization_c()
        candidates = feature_candidates(c_value)
        based_on = f"Best regularization value C={c_value:g}"
    else:
        candidates = joint_candidates()
        based_on = "Feature-round best: 50k word unigrams, sublinear TF, C=30"
    manifest = {
        "round": round_name, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": main.RANDOM_STATE, "cv_folds": main.N_SPLITS, "jobs": jobs,
        "based_on": based_on, "train_sha256": sha256(train_path),
        "initial_splits_sha256": sha256(INITIAL_DIR / "cv_splits.csv"),
        "source_sha256": sha256(Path(__file__)), "candidates": candidates,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    shutil.copy2(Path(__file__), output_dir / "source_snapshot.py")
    results = run_candidates(candidates, X, y, splits, output_dir, jobs)
    manifest.update({
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_candidate_id": results.iloc[0]["candidate_id"],
        "best_macro_f1_mean": results.iloc[0]["macro_f1_mean"],
    })
    write_json(output_dir / "run_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=("regularization", "features", "joint"), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    if args.jobs == 0 or args.jobs < -1:
        raise ValueError("--jobs must be -1 or a positive integer.")
    log_dir = TUNING_DIR / args.round
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "run.log", encoding="utf-8")],
    )
    with threadpool_limits(limits=1), joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        run_round(args.round, args.jobs)
    LOGGER.info("Tuning round completed: %s", args.round)


if __name__ == "__main__":
    cli()
