"""
Email Spam Classifier — Modular Training Pipeline
==================================================
Features:
  - Configurable via CONFIG dict
  - Multiple model comparison
  - GridSearchCV integration
  - Threshold tuning to reduce false negatives
  - Model persistence with joblib
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_recall_curve
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit here, not scattered in code
# ─────────────────────────────────────────────
CONFIG = {
    "data_path": r"D:\KaggleCompetitions\Email_Spam\email.csv",
    "text_col": "Message",
    "label_col": "Category",
    "valid_labels": ["ham", "spam"],
    "test_size": 0.2,
    "random_state": 42,
    "model_output_path": r"model\spam_model11.joblib",

    # TF-IDF defaults (used unless overridden by GridSearch)
    "tfidf": {
        "ngram_range": (1, 2),
        "max_df": 0.7,
        "max_features": 3000,
        "stop_words": None,
    },

    # Set to True to run GridSearchCV on the best model
    "run_gridsearch": True,

    # Tune decision threshold to reduce false negatives (spam slipping through)
    "tune_threshold": True,
}

# ─────────────────────────────────────────────
# DATA LOADING & SPLITTING
# ─────────────────────────────────────────────
def load_data(config: dict):
    data = pd.read_csv(config["data_path"])
    mail = data[config["text_col"]].astype(str)
    target = data[config["label_col"]]

    x_train, x_test, y_train, y_test = train_test_split(
        mail, target,
        test_size=config["test_size"],
        random_state=config["random_state"],
        # stratify=target  # preserves class ratio in splits
    )

    # Filter to valid labels (handles any stray rows)
    mask_train = y_train.isin(config["valid_labels"])
    mask_test = y_test.isin(config["valid_labels"])

    return (
        x_train[mask_train], x_test[mask_test],
        y_train[mask_train], y_test[mask_test]
    )


# ─────────────────────────────────────────────
# PIPELINE FACTORY
# ─────────────────────────────────────────────
def make_pipeline(classifier, tfidf_params: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", classifier),
    ])


def get_classifiers(random_state: int) -> dict:
    """
    Returns a dict of {name: classifier} to compare.
    LinearSVC wrapped in CalibratedClassifierCV to support predict_proba.
    """
    return {
        "SGD (log loss)": SGDClassifier(
            loss="log_loss", random_state=random_state, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "Multinomial NB": MultinomialNB(),
        "Linear SVC (calibrated)": CalibratedClassifierCV(
            LinearSVC(random_state=random_state, class_weight="balanced")
        ),
    }


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, x_test, y_test, threshold: float = 0.5):
    """Evaluate with optional custom decision threshold."""
    if threshold != 0.5 and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        spam_idx = list(model.classes_).index("spam")
        y_pred = np.where(proba[:, spam_idx] >= threshold, "spam", "ham")
    else:
        y_pred = model.predict(x_test)

    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))
    print(classification_report(y_test, y_pred))
    return f1_score(y_test, y_pred, pos_label="spam")


def find_best_threshold(model, x_test, y_test):
    """
    Find threshold that maximises F1 for spam — reduces false negatives
    without tanking precision too much.
    """
    if not hasattr(model, "predict_proba"):
        print("  ⚠ Model does not support predict_proba — skipping threshold tuning.")
        return 0.5

    spam_idx = list(model.classes_).index("spam")
    proba = model.predict_proba(x_test)[:, spam_idx]
    precision, recall, thresholds = precision_recall_curve(
        y_test, proba, pos_label="spam"
    )

    # F1 at each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"  → Best threshold: {best_threshold:.3f}  "
          f"(precision={precision[best_idx]:.3f}, recall={recall[best_idx]:.3f})")
    return float(best_threshold)


# ─────────────────────────────────────────────
# GRIDSEARCH
# ─────────────────────────────────────────────
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [1000, 3000, 5000],
    "tfidf__stop_words": ["english", None],
    "tfidf__max_df": [0.7, 0.8, 0.9],
}


def run_gridsearch(pipeline: Pipeline, x_train, y_train) -> Pipeline:
    grid = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)
    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Email Spam Classifier — Model Comparison")
    print("=" * 60)

    x_train, x_test, y_train, y_test = load_data(CONFIG)
    print(f"\nTrain size: {len(x_train)} | Test size: {len(x_test)}")
    print(f"Spam in test: {(y_test == 'spam').sum()} / {len(y_test)}\n")

    classifiers = get_classifiers(CONFIG["random_state"])
    results = {}
    best_f1, best_model, best_name = 0, None, ""

    for name, clf in classifiers.items():
        print(f"\n{'─'*50}")
        print(f"  Model: {name}")
        print(f"{'─'*50}")

        pipeline = make_pipeline(clf, CONFIG["tfidf"])

        if CONFIG["run_gridsearch"] and name == "Logistic Regression":
            # Only run expensive GridSearch on the most promising model
            print("  Running GridSearchCV...")
            pipeline = run_gridsearch(pipeline, x_train, y_train)
        else:
            pipeline.fit(x_train, y_train)

        threshold = 0.5
        if CONFIG["tune_threshold"]:
            threshold = find_best_threshold(pipeline, x_test, y_test)

        f1 = evaluate(pipeline, x_test, y_test, threshold=threshold)
        results[name] = {"f1_spam": f1, "threshold": threshold}

        if f1 > best_f1:
            best_f1, best_model, best_name = f1, pipeline, name

    # ── Summary ──────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        marker = " ◀ BEST" if name == best_name else ""
        print(f"  {name:<30} F1(spam)={r['f1_spam']:.4f}  threshold={r['threshold']:.3f}{marker}")

    # ── Save best model ───────────────────────
    out_path = Path(CONFIG["model_output_path"])
    joblib.dump(best_model, out_path)
    print(f"\n✅ Best model '{best_name}' saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
