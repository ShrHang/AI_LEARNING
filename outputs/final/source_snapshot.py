"""Select the best completed round, create diagnostics, and fit the final model."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "project1-matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_predict
from threadpoolctl import threadpool_limits

import main
import tune


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "final"
ROUND_DIRS = [
    ("initial", PROJECT_DIR / "outputs" / "restarted"),
    ("regularization", PROJECT_DIR / "outputs" / "tuning" / "regularization"),
    ("features", PROJECT_DIR / "outputs" / "tuning" / "features"),
    ("joint", PROJECT_DIR / "outputs" / "tuning" / "joint"),
    ("convergence", PROJECT_DIR / "outputs" / "tuning" / "convergence"),
]
LOGGER = logging.getLogger("project1-final")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=tune.json_default) + "\n",
        encoding="utf-8",
    )


def generic_initial_candidate(candidate: dict) -> dict:
    config = candidate["feature_config"]
    return {
        "candidate_id": candidate["candidate_id"],
        "vectorizer": {
            "analyzer": "word", "ngram_range": [1, 1],
            "max_features": config["max_features"], "min_df": 1,
            "sublinear_tf": config["sublinear_tf"],
        },
        "classifier": candidate["model"],
        "C": float(candidate["parameters"]["C"]),
    }


def collect_round_winners(train_path: Path) -> tuple[pd.DataFrame, dict]:
    train_hash = sha256(train_path)
    rows, candidates = [], {}
    for round_name, directory in ROUND_DIRS:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] not in {"comparison_complete", "complete"}:
            raise ValueError(f"Round is incomplete: {round_name}")
        if manifest["train_sha256"] != train_hash:
            raise ValueError(f"Training data hash mismatch in round: {round_name}")
        selected = json.loads((directory / "selected_config.json").read_text(encoding="utf-8"))
        candidate = selected["candidate"]
        if round_name == "initial":
            candidate = generic_initial_candidate(candidate)
        metrics = selected["cv_metrics"]
        rows.append({
            "round": round_name, "candidate_id": candidate["candidate_id"],
            "macro_f1_mean": float(metrics["macro_f1_mean"]),
            "macro_f1_std": float(metrics["macro_f1_std"]),
            "accuracy_mean": float(metrics["accuracy_mean"]),
            "accuracy_std": float(metrics["accuracy_std"]),
        })
        candidates[candidate["candidate_id"]] = candidate
    summary = pd.DataFrame(rows).sort_values(
        ["macro_f1_mean", "macro_f1_std", "candidate_id"], ascending=[False, True, True],
    ).reset_index(drop=True)
    best = candidates[summary.iloc[0]["candidate_id"]]
    return summary, best


def save_progress_plot(summary: pd.DataFrame) -> None:
    chronological = summary.set_index("round").loc[[name for name, _ in ROUND_DIRS]].reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        chronological["round"], chronological["macro_f1_mean"],
        yerr=chronological["macro_f1_std"], marker="o", capsize=4, color="#4E79A7",
    )
    ax.set_ylabel("Validation Macro-F1 (mean ± fold standard deviation)")
    ax.set_xlabel("Tuning round")
    ax.set_title("Best candidate from each tuning round")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "tuning_progress.png", dpi=180)
    plt.close(fig)


def create_oof_diagnostics(
    X: np.ndarray, y: np.ndarray, candidate: dict,
    splits: list[tuple[np.ndarray, np.ndarray]], jobs: int,
) -> None:
    LOGGER.info("Generating final candidate out-of-fold predictions")
    predicted = cross_val_predict(tune.build_pipeline(candidate), X, y, cv=splits, n_jobs=jobs)
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, valid_ids) in enumerate(splits, start=1):
        fold_ids[valid_ids] = fold
    pd.DataFrame({
        "row_index": np.arange(len(y)), "fold": fold_ids,
        "true_label": y, "predicted_label": predicted,
    }).to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    report = classification_report(
        y, predicted, labels=main.LABELS, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / "classification_report.csv")
    errors = pd.DataFrame({
        "row_index": np.arange(len(y)), "true_label": y, "predicted_label": predicted,
        "text_excerpt": [re.sub(r"\s+", " ", text)[:500] for text in X],
    })
    errors.loc[y != predicted].to_csv(OUTPUT_DIR / "misclassified_examples.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y, predicted, labels=main.LABELS, cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Final candidate: out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def run(jobs: int = 2) -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT_DIR}")
    for directory in (OUTPUT_DIR, OUTPUT_DIR / "figures", OUTPUT_DIR / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8")],
    )
    train_path = PROJECT_DIR / "data-Project1" / "train_data.csv"
    test_path = PROJECT_DIR / "data-Project1" / "test_data_unlabeled.csv"
    X, y = main.load_training_data(train_path)
    splits = tune.load_fixed_splits(X, y)
    summary, candidate = collect_round_winners(train_path)
    summary.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)
    write_json(OUTPUT_DIR / "selected_config.json", {
        "candidate": candidate,
        "selection_rule": "best mean Macro-F1 across completed tuning rounds",
        "cv_metrics": summary.iloc[0].to_dict(),
        "stopping_rule": "stop after a completed round improves mean Macro-F1 by less than 0.001",
    })
    save_progress_plot(summary)
    create_oof_diagnostics(X, y, candidate, splits, jobs)

    # The test CSV is first loaded after selection and CV diagnostics are fixed.
    X_test = main.load_test_data(test_path)
    pipeline = tune.build_pipeline(candidate)
    LOGGER.info("Fitting final pipeline on all %d labeled rows", len(y))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X_test)
    if len(predictions) != len(X_test) or not np.isin(predictions, main.LABELS).all():
        raise ValueError("Invalid final predictions.")
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "predictions.csv", index=False, header=False)
    joblib.dump(pipeline, OUTPUT_DIR / "models" / "final_pipeline.joblib")
    manifest = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": main.RANDOM_STATE, "cv_folds": main.N_SPLITS, "jobs": jobs,
        "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
        "main_source_sha256": sha256(Path(main.__file__)),
        "tune_source_sha256": sha256(Path(tune.__file__)),
        "finalize_source_sha256": sha256(Path(__file__)),
        "selected_candidate": candidate, "training_samples": len(X), "test_samples": len(X_test),
        "actual_features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "prediction_sha256": sha256(OUTPUT_DIR / "predictions.csv"),
    }
    write_json(OUTPUT_DIR / "run_manifest.json", manifest)
    shutil.copy2(Path(__file__), OUTPUT_DIR / "source_snapshot.py")
    LOGGER.info("Saved %d final predictions", len(predictions))


if __name__ == "__main__":
    with threadpool_limits(limits=1), joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        run(jobs=2)
