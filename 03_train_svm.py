import json
import platform
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, confusion_matrix
)
import joblib
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_ANALYZER,
    TFIDF_SUBLINEAR_TF, SVM_C_VALUES
)


def make_vectorizer():
    return TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )


def make_svm(c):
    return LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=RANDOM_SEED)


def classification_metrics(y_true, y_pred):
    """Shared metric set (malicious = positive class)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_benign": float(f1_score(y_true, y_pred, pos_label=0)),
        "f1_malicious": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "precision_benign": float(precision_score(y_true, y_pred, pos_label=0)),
        "precision_malicious": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "recall_benign": float(recall_score(y_true, y_pred, pos_label=0)),
        "recall_malicious": float(recall_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
    }


def train_svm():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    train_df["script_text"] = train_df["script_text"].fillna("")
    val_df["script_text"] = val_df["script_text"].fillna("")
    test_df["script_text"] = test_df["script_text"].fillna("")

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # --- Hyperparameter search (not part of the reported training time) ---
    print("Fitting TF-IDF vectorizer for C search...")
    t0 = time.perf_counter()
    search_vec = make_vectorizer()
    X_train = search_vec.fit_transform(train_df["script_text"])
    X_val = search_vec.transform(val_df["script_text"])
    print(f"  shape: {X_train.shape}")

    print("\nTuning SVM regularization on validation set...")
    best_c = None
    best_f1 = -1
    c_search = {}

    for c in SVM_C_VALUES:
        svm = make_svm(c)
        svm.fit(X_train, y_train)
        val_preds = svm.predict(X_val)
        f1 = f1_score(y_val, val_preds, average="macro")
        c_search[str(c)] = float(f1)
        print(f"  C={c}: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
    search_time = time.perf_counter() - t0

    print(f"\nBest C={best_c} (val F1={best_f1:.4f})")

    # --- Training, timed end-to-end: TF-IDF fit + SVM fit from raw text ---
    print("Training final model end-to-end (TF-IDF fit + SVM fit)...")
    t0 = time.perf_counter()
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform(train_df["script_text"])
    t_vec = time.perf_counter()
    final_svm = make_svm(best_c)
    final_svm.fit(X_train, y_train)
    t_end = time.perf_counter()
    train_time = t_end - t0
    tfidf_fit_time = t_vec - t0
    svm_fit_time = t_end - t_vec
    print(f"  train time: {train_time:.2f}s (TF-IDF {tfidf_fit_time:.2f}s + SVM {svm_fit_time:.2f}s)")

    # Calibrated copy only provides probabilities for the ROC curve in 04_evaluate.
    t0 = time.perf_counter()
    calibrated_svm = CalibratedClassifierCV(make_svm(best_c), cv=3, method="sigmoid")
    calibrated_svm.fit(X_train, y_train)
    calibration_time = time.perf_counter() - t0

    # --- Classification, timed end-to-end: TF-IDF transform + predict ---
    print("Evaluating on test set...")
    test_texts = test_df["script_text"].tolist()
    t0 = time.perf_counter()
    X_test = vectorizer.transform(test_texts)
    t_vec = time.perf_counter()
    test_preds = final_svm.predict(X_test)
    t_end = time.perf_counter()
    inference_time = t_end - t0
    transform_time = t_vec - t0
    predict_only_time = t_end - t_vec

    test_probs = calibrated_svm.predict_proba(X_test)[:, 1]
    test_scores = final_svm.decision_function(X_test)

    nnz = X_test.getnnz(axis=1)
    n_test = len(y_test)

    metrics = {
        "model": "TF-IDF + SVM",
        "seed": RANDOM_SEED,
        "best_C": best_c,
        "val_f1_by_C": c_search,
        **classification_metrics(y_test, test_preds),
        "train_time_seconds": round(train_time, 3),
        "train_time_breakdown_seconds": {
            "tfidf_fit_transform": round(tfidf_fit_time, 3),
            "svm_fit": round(svm_fit_time, 3),
        },
        "c_search_time_seconds": round(search_time, 3),
        "calibration_time_seconds": round(calibration_time, 3),
        "inference_time_seconds": round(inference_time, 4),
        "inference_per_script_ms": round(inference_time / n_test * 1000, 4),
        "inference_breakdown_seconds": {
            "tfidf_transform": round(transform_time, 4),
            "svm_predict_only": round(predict_only_time, 6),
        },
        "predict_only_per_script_ms": round(predict_only_time / n_test * 1000, 6),
        "n_test_scripts": n_test,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "nonzero_ngrams_per_test_script": {
            "median": float(np.median(nnz)),
            "mean": float(np.mean(nnz)),
            "min": int(nnz.min()),
            "max": int(nnz.max()),
        },
        "hardware": {"device": "cpu", "processor": platform.processor(), "machine": platform.machine()},
    }

    print(f"\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(final_svm, MODELS_DIR / "svm_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    preds_df = pd.DataFrame({
        "source_file": test_df["source_file"],
        "text_sha256": test_df["text_sha256"],
        "true_label": y_test,
        "predicted_label": test_preds,
        "decision_score": test_scores,
        "malicious_probability": test_probs,
        "nonzero_ngrams": nnz,
    })
    preds_df.to_csv(RESULTS_DIR / "svm_predictions.csv", index=False)

    with open(RESULTS_DIR / "svm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR}")
    print(f"Saved predictions and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    train_svm()
