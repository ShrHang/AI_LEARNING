"""Compare three text classifiers using identical stratified five-fold splits.

No experiment runs on import. Use --stage compare, final, or all from the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Support machines without a display or a writable home cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "project1-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
N_SPLITS = 5
LABELS = list(range(10))
LOGGER = logging.getLogger("project1")

# Change one feature factor at a time between these configurations.
TFIDF_CONFIGS = [
    {"config_id": "unigram_5k", "max_features": 5_000, "sublinear_tf": False},
    {"config_id": "unigram_20k", "max_features": 20_000, "sublinear_tf": False},
    {"config_id": "unigram_20k_sublinear", "max_features": 20_000, "sublinear_tf": True},
]
MODEL_GRIDS = {
    "MultinomialNB": {"alpha": [0.1, 0.5, 1.0]},
    "SVC": {"C": [0.1, 1.0, 10.0]},
    "LogisticRegression": {"C": [0.1, 1.0, 10.0, 20.0]},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> object:
    """Convert non-standard parameter values into stable JSON representations."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["text", "target"]:
        raise ValueError("Training CSV must contain exactly text,target columns.")
    if frame.isna().any().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Training data contains missing values or blank text.")
    labels = pd.to_numeric(frame["target"], errors="raise")
    if not labels.isin(LABELS).all() or set(labels.unique()) != set(LABELS):
        raise ValueError("Training labels must be integer class IDs 0 through 9.")
    if labels.value_counts().min() < N_SPLITS:
        raise ValueError("Each class needs at least five samples for stratified CV.")
    return frame["text"].astype(str).to_numpy(), labels.to_numpy(dtype=int)


def load_test_data(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["text"]:
        raise ValueError("Test CSV must contain exactly the text column.")
    if frame.empty or frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Test data is empty or contains missing/blank text.")
    return frame["text"].astype(str).to_numpy()


def make_candidates() -> list[dict]:
    candidates = []
    for config in TFIDF_CONFIGS:
        for model_name, grid in MODEL_GRIDS.items():
            for parameter, values in grid.items():
                for value in values:
                    candidates.append({
                        "candidate_id": f"{config['config_id']}__{model_name}__{parameter}_{value:g}",
                        "feature_config": dict(config),
                        "model": model_name,
                        "parameters": {parameter: value},
                    })
    return candidates


def build_pipeline(candidate: dict) -> Pipeline:
    config = candidate["feature_config"]
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        sublinear_tf=config["sublinear_tf"],
        ngram_range=(1, 1), min_df=1, lowercase=True, stop_words=None, norm="l2",
    )
    parameters = candidate["parameters"]
    if candidate["model"] == "MultinomialNB":
        classifier = MultinomialNB(**parameters)
    elif candidate["model"] == "SVC":
        classifier = SVC(kernel="linear", random_state=RANDOM_STATE, **parameters)
    elif candidate["model"] == "LogisticRegression":
        classifier = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE, **parameters)
    else:
        raise ValueError(f"Unsupported classifier: {candidate['model']}")
    # Fit vocabulary, IDF, and classifier using only each fold's training rows.
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def make_splits(X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y))


def describe_environment() -> dict:
    return {
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
        },
    }


def plot_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    ordered = results.sort_values("macro_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.3 * len(ordered))))
    colors = {"MultinomialNB": "#59A14F", "SVC": "#F28E2B", "LogisticRegression": "#4E79A7"}
    ax.barh(
        ordered["candidate_id"], ordered["macro_f1_mean"],
        xerr=ordered["macro_f1_std"], color=ordered["model"].map(colors), capsize=2,
    )
    ax.set_xlabel("Validation Macro-F1 (mean ± fold standard deviation)")
    ax.set_title("Same stratified five-fold splits for every candidate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "cv_comparison.png", dpi=160)
    plt.close(fig)


def compare_candidates(
    X: np.ndarray, y: np.ndarray, candidates: list[dict],
    splits: list[tuple[np.ndarray, np.ndarray]], output_dir: Path, jobs: int,
) -> tuple[pd.DataFrame, dict]:
    summaries, fold_rows = [], []
    for index, candidate in enumerate(candidates, start=1):
        LOGGER.info("Candidate %d/%d: %s", index, len(candidates), candidate["candidate_id"])
        start = perf_counter()
        scores = cross_validate(
            build_pipeline(candidate), X, y, cv=splits,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=jobs, return_train_score=True, error_score="raise",
        )
        row = {
            "candidate_id": candidate["candidate_id"],
            "config_id": candidate["feature_config"]["config_id"], "model": candidate["model"],
            "parameters": json.dumps(candidate["parameters"], sort_keys=True),
            "max_features": candidate["feature_config"]["max_features"],
            "sublinear_tf": candidate["feature_config"]["sublinear_tf"],
            "wall_time_seconds": perf_counter() - start,
            "fit_time_mean": float(scores["fit_time"].mean()),
        }
        for metric in ("accuracy", "macro_f1"):
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"train_{metric}_mean"] = float(scores[f"train_{metric}"].mean())
        summaries.append(row)
        for fold, (train_ids, valid_ids) in enumerate(splits):
            fold_rows.append({
                "candidate_id": candidate["candidate_id"], "fold": fold + 1,
                "train_samples": len(train_ids), "validation_samples": len(valid_ids),
                "train_accuracy": scores["train_accuracy"][fold],
                "validation_accuracy": scores["test_accuracy"][fold],
                "train_macro_f1": scores["train_macro_f1"][fold],
                "validation_macro_f1": scores["test_macro_f1"][fold],
                "fit_time_seconds": scores["fit_time"][fold], "score_time_seconds": scores["score_time"][fold],
            })
        # Keep completed candidates if a later fit fails.
        pd.DataFrame(summaries).to_csv(output_dir / "cv_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "cv_fold_results.csv", index=False)
        LOGGER.info("Macro-F1 %.4f ± %.4f; Accuracy %.4f", row["macro_f1_mean"], row["macro_f1_std"], row["accuracy_mean"])

    # Declare a stable selection rule instead of choosing it after seeing scores.
    results = pd.DataFrame(summaries).sort_values(
        ["macro_f1_mean", "macro_f1_std", "candidate_id"], ascending=[False, True, True],
    ).reset_index(drop=True)
    results.to_csv(output_dir / "cv_results.csv", index=False)
    best_id = results.iloc[0]["candidate_id"]
    best = next(candidate for candidate in candidates if candidate["candidate_id"] == best_id)
    write_json(output_dir / "selected_config.json", {
        "candidate": best,
        "selection_rule": "highest macro_f1_mean, then lowest macro_f1_std, then candidate_id",
        "cv_metrics": results.iloc[0].to_dict(),
        "evaluation_note": "Used for model selection; not an independent test estimate.",
    })
    plot_comparison(results, output_dir)
    return results, best


def save_diagnostics(
    X: np.ndarray, y: np.ndarray, candidate: dict,
    splits: list[tuple[np.ndarray, np.ndarray]], output_dir: Path, jobs: int,
) -> None:
    LOGGER.info("Generating selected candidate's out-of-fold diagnostics")
    predicted = cross_val_predict(build_pipeline(candidate), X, y, cv=splits, n_jobs=jobs)
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, valid_ids) in enumerate(splits, start=1):
        fold_ids[valid_ids] = fold
    pd.DataFrame({
        "row_index": np.arange(len(y)), "fold": fold_ids, "true_label": y, "predicted_label": predicted,
    }).to_csv(output_dir / "oof_predictions.csv", index=False)
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    errors = pd.DataFrame({
        "row_index": np.arange(len(y)), "true_label": y, "predicted_label": predicted,
        "text_excerpt": [re.sub(r"\s+", " ", text)[:500] for text in X],
    })
    errors.loc[y != predicted].to_csv(output_dir / "misclassified_examples.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, predicted, labels=LABELS, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Selected candidate: out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def run_comparison(train_path: Path, output_dir: Path, jobs: int) -> dict:
    X, y = load_training_data(train_path)
    splits = make_splits(X, y)
    candidates = make_candidates()
    fold_ids = np.empty(len(y), dtype=int)
    for fold, (_, valid_ids) in enumerate(splits, start=1):
        fold_ids[valid_ids] = fold
    pd.DataFrame({"row_index": np.arange(len(y)), "target": y, "validation_fold": fold_ids}).to_csv(
        output_dir / "cv_splits.csv", index=False,
    )
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running_comparison",
        "random_state": RANDOM_STATE, "cv_folds": N_SPLITS, "jobs": jobs,
        "std_ddof": 0, "selection_metric": "macro_f1",
        "train_sha256": file_sha256(train_path), "source_sha256": file_sha256(Path(__file__)),
        "training_samples": len(y), "class_counts": {str(label): int((y == label).sum()) for label in LABELS},
        "candidates": candidates,
        "pipeline_parameters": {candidate["candidate_id"]: {
            key: value for key, value in build_pipeline(candidate).get_params().items() if "__" in key
        } for candidate in candidates},
        "environment": describe_environment(),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    LOGGER.info("%d labeled rows; %d candidates; %d folds", len(y), len(candidates), N_SPLITS)
    _, selected = compare_candidates(X, y, candidates, splits, output_dir, jobs)
    save_diagnostics(X, y, selected, splits, output_dir, jobs)
    manifest["status"] = "comparison_complete"
    manifest["comparison_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["selected_config_sha256"] = file_sha256(output_dir / "selected_config.json")
    write_json(output_dir / "run_manifest.json", manifest)
    return selected


def run_final(train_path: Path, test_path: Path, output_dir: Path) -> None:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in {"comparison_complete", "complete"}:
        raise ValueError("Final training requires a successfully completed comparison.")
    if manifest["train_sha256"] != file_sha256(train_path):
        raise ValueError("Training data changed since comparison; rerun the comparison.")
    if manifest["source_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Experiment source changed since comparison; rerun the comparison.")
    if manifest["selected_config_sha256"] != file_sha256(output_dir / "selected_config.json"):
        raise ValueError("Selected configuration changed since comparison.")
    if manifest["environment"]["packages"] != describe_environment()["packages"]:
        raise ValueError("Package versions changed since comparison; use the original environment.")
    selected = json.loads((output_dir / "selected_config.json").read_text(encoding="utf-8"))["candidate"]
    X, y = load_training_data(train_path)
    # Load test text only after successful model selection.
    X_test = load_test_data(test_path)
    pipeline = build_pipeline(selected)
    LOGGER.info("Refitting %s on all %d labeled rows", selected["candidate_id"], len(y))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X_test)
    if len(predictions) != len(X_test) or not np.isin(predictions, LABELS).all():
        raise ValueError("Invalid test predictions.")
    pd.DataFrame(predictions).to_csv(output_dir / "predictions.csv", index=False, header=False)
    joblib.dump(pipeline, output_dir / "models" / "final_pipeline.joblib")
    manifest.update({
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_sha256": file_sha256(test_path), "test_samples": len(X_test),
        "final_actual_features": len(pipeline.named_steps["tfidf"].vocabulary_),
        "prediction_sha256": file_sha256(output_dir / "predictions.csv"),
    })
    write_json(output_dir / "run_manifest.json", manifest)
    LOGGER.info("Saved %d predictions and final model", len(predictions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compare", "final", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data-Project1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "restarted")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel folds (default: 1; -1: all CPUs).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs == 0 or args.jobs < -1:
        raise ValueError("--jobs must be -1 or a positive integer.")
    output_dir = args.output_dir.resolve()
    if args.stage in {"compare", "all"} and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty. Choose a new --output-dir to preserve previous runs.")
    if args.stage == "final" and not (output_dir / "run_manifest.json").is_file():
        raise FileNotFoundError("Run --stage compare first with the same --output-dir.")
    for directory in (output_dir, output_dir / "figures", output_dir / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")],
    )
    logging.captureWarnings(True)
    try:
        # Limit numerical-library threads in the parent and parallel workers.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1), joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            if args.stage in {"compare", "all"}:
                run_comparison(args.data_dir / "train_data.csv", output_dir, args.jobs)
            if args.stage in {"final", "all"}:
                run_final(args.data_dir / "train_data.csv", args.data_dir / "test_data_unlabeled.csv", output_dir)
        LOGGER.info("Requested stage completed: %s", args.stage)
    except Exception as error:
        LOGGER.exception("Experiment failed")
        write_json(output_dir / "failure.json", {
            "stage": args.stage, "error_type": type(error).__name__, "error": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
