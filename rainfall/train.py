"""Run from the repository root: python -m rainfall.train."""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import SHA256, SOURCE, STATIONS, date_folds, download, features, load, partitions


def preprocessing(X):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f1)])


def scores(y, probability, threshold):
    predicted = probability >= threshold
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted)),
            "f1": float(f1_score(y, predicted)),
            "threshold": threshold}


def run(data_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load(download(data_path))
    X, y = features(frame), frame.RainTomorrow.eq("Yes").astype(int)
    train, valid, test = partitions(frame.Date)
    folds = list(date_folds(frame.Date.iloc[train]))
    candidates = {
        "logistic_regression": (LogisticRegression(max_iter=2500, random_state=42),
            {"model__C": [.1, 1, 10], "model__class_weight": [None, "balanced"]}),
        "random_forest": (RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1,
            class_weight="balanced_subsample"),
            {"model__max_depth": [8, None], "model__min_samples_leaf": [4, 12]}),
    }
    fitted, selection = {}, []
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessing(X)), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, cv=folds, scoring="average_precision", n_jobs=2,
                              error_score="raise", refit=True)
        search.fit(X.iloc[train], y.iloc[train])
        probability = search.predict_proba(X.iloc[valid])[:, 1]
        fitted[name] = search.best_estimator_
        selection.append({"model": name, "cv_average_precision": search.best_score_,
            "validation_average_precision": average_precision_score(y.iloc[valid], probability),
            "best_parameters": search.best_params_})
    selection.sort(key=lambda row: row["validation_average_precision"], reverse=True)
    winner = selection[0]["model"]
    estimator = fitted[winner]
    # Freeze model and decision threshold before reading the test labels for evaluation.
    threshold = choose_threshold(y.iloc[valid], estimator.predict_proba(X.iloc[valid])[:, 1])
    test_probability = estimator.predict_proba(X.iloc[test])[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(X.iloc[train], y.iloc[train])
    baseline_probability = dummy.predict_proba(X.iloc[test])[:, 1]
    summary = {"selected_model": winner, "selection": selection, "test": scores(y.iloc[test], test_probability, threshold),
        "baseline_test": scores(y.iloc[test], baseline_probability, .5),
        "partitions": {name: {"rows": len(index), "dates": frame.Date.iloc[index].nunique(),
            "start": str(frame.Date.iloc[index].min().date()), "end": str(frame.Date.iloc[index].max().date()),
            "rain_rate": float(y.iloc[index].mean())}
            for name, index in zip(("train", "validation", "test"), (train, valid, test))},
        "source": SOURCE, "sha256": SHA256, "stations": STATIONS, "seed": 42,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__},
        "protocol": "Date-ordered 70/15/15 split; training-only expanding-date CV; validation selects model and F1 threshold; no validation refit."}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(selection).to_csv(output / "model_selection.csv", index=False)
    predictions = frame.loc[test, ["Date", "Location"]].copy()
    predictions["observed_rain_tomorrow"] = y.iloc[test].to_numpy()
    predictions["probability"] = test_probability
    predictions["predicted_rain_tomorrow"] = (test_probability >= threshold).astype(int)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    precision, recall, _ = precision_recall_curve(y.iloc[test], test_probability)
    axes[0].plot(recall, precision, color="#007f86", label=winner)
    axes[0].axhline(y.iloc[test].mean(), color="gray", ls="--", label="Test prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Untouched test: precision–recall")
    axes[0].legend(fontsize=8)
    ConfusionMatrixDisplay(confusion_matrix(y.iloc[test], test_probability >= threshold),
        display_labels=["No rain", "Rain"]).plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"Validation-selected threshold: {threshold:.3f}")
    fraction, mean = calibration_curve(y.iloc[test], test_probability, n_bins=8, strategy="quantile")
    axes[2].plot(mean, fraction, "o-", color="#007f86")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed rain fraction", title="Reliability (descriptive)")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"model": winner, "test": summary["test"], "rows": len(frame)}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/weatherAUS.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.data, args.output)
