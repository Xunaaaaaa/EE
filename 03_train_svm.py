import json
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score
)
import joblib
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_ANALYZER,
    TFIDF_SUBLINEAR_TF, SVM_C_VALUES
)

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

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )

    t0 = time.time()
    X_train = vectorizer.fit_transform(train_df["script_text"])
    tfidf_time = time.time() - t0
    print(f"  TF-IDF fit_transform: {tfidf_time:.1f}s, shape: {X_train.shape}")

    X_val = vectorizer.transform(val_df["script_text"])
    X_test = vectorizer.transform(test_df["script_text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print("\nTuning SVM regularization on validation set...")
    best_c = None
    best_f1 = -1

    for c in SVM_C_VALUES:
        svm = LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=42)
        svm.fit(X_train, y_train)
        val_preds = svm.predict(X_val)
        f1 = f1_score(y_val, val_preds, average="macro")
        print(f"  C={c}: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_c = c

    print(f"\nBest C={best_c} (val F1={best_f1:.4f})")

    print("Training final model...")
    t0 = time.time()
    final_svm = LinearSVC(C=best_c, class_weight="balanced", max_iter=10000, random_state=42)
    final_svm.fit(X_train, y_train)
    train_time = time.time() - t0

    calibrated_svm = CalibratedClassifierCV(final_svm, cv=3, method="sigmoid")
    calibrated_svm.fit(X_train, y_train)

    print("Evaluating on test set...")
    t0 = time.time()
    test_preds = final_svm.predict(X_test)
    inference_time = time.time() - t0

    test_probs = calibrated_svm.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "TF-IDF + SVM",
        "best_C": best_c,
        "f1_macro": float(f1_score(y_test, test_preds, average="macro")),
        "f1_benign": float(f1_score(y_test, test_preds, pos_label=0)),
        "f1_malicious": float(f1_score(y_test, test_preds, pos_label=1)),
        "precision_macro": float(precision_score(y_test, test_preds, average="macro")),
        "recall_macro": float(recall_score(y_test, test_preds, average="macro")),
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_preds)),
        "train_time_seconds": round(train_time, 2),
        "inference_time_seconds": round(inference_time, 4),
        "inference_per_script_ms": round((inference_time / len(y_test)) * 1000, 4),
    }

    print(f"\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(final_svm, MODELS_DIR / "svm_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    preds_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": test_preds,
        "malicious_probability": test_probs,
    })
    preds_df.to_csv(RESULTS_DIR / "svm_predictions.csv", index=False)

    with open(RESULTS_DIR / "svm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR}")
    print(f"Saved predictions and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    train_svm()
