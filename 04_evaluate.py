import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import RESULTS_DIR


def load_results():
    svm_preds = pd.read_csv(RESULTS_DIR / "svm_predictions.csv")
    codebert_preds = pd.read_csv(RESULTS_DIR / "codebert_predictions.csv")

    with open(RESULTS_DIR / "svm_metrics.json") as f:
        svm_metrics = json.load(f)
    with open(RESULTS_DIR / "codebert_metrics.json") as f:
        codebert_metrics = json.load(f)

    return svm_preds, codebert_preds, svm_metrics, codebert_metrics


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_f1_comparison(svm_metrics, codebert_metrics):
    labels = ["F1 Macro", "F1 Benign", "F1 Malicious"]
    svm_vals = [svm_metrics["f1_macro"], svm_metrics["f1_benign"], svm_metrics["f1_malicious"]]
    cb_vals = [codebert_metrics["f1_macro"], codebert_metrics["f1_benign"], codebert_metrics["f1_malicious"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, svm_vals, width, label="TF-IDF + SVM", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, cb_vals, width, label="CodeBERT", color="#DD8452")

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison: TF-IDF + SVM vs CodeBERT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved f1_comparison.png")


def plot_roc_curves(svm_preds, codebert_preds):
    fig, ax = plt.subplots(figsize=(7, 6))

    fpr_svm, tpr_svm, _ = roc_curve(svm_preds["true_label"], svm_preds["malicious_probability"])
    auc_svm = auc(fpr_svm, tpr_svm)
    ax.plot(fpr_svm, tpr_svm, label=f"TF-IDF + SVM (AUC={auc_svm:.3f})", color="#4C72B0")

    fpr_cb, tpr_cb, _ = roc_curve(codebert_preds["true_label"], codebert_preds["malicious_probability"])
    auc_cb = auc(fpr_cb, tpr_cb)
    ax.plot(fpr_cb, tpr_cb, label=f"CodeBERT (AUC={auc_cb:.3f})", color="#DD8452")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: TF-IDF + SVM vs CodeBERT")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved roc_curves.png")


def create_summary_table(svm_metrics, codebert_metrics):
    keys = ["f1_macro", "f1_benign", "f1_malicious", "precision_macro",
            "recall_macro", "accuracy", "balanced_accuracy",
            "train_time_seconds", "inference_per_script_ms"]
    rows = []
    for k in keys:
        rows.append({
            "Metric": k,
            "TF-IDF + SVM": svm_metrics.get(k, ""),
            "CodeBERT": codebert_metrics.get(k, ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print("  Saved metrics_summary.csv")
    print(f"\n{df.to_string(index=False)}")


def main():
    print("Loading results...")
    svm_preds, codebert_preds, svm_metrics, codebert_metrics = load_results()

    print("\nGenerating charts...")
    plot_confusion_matrix(
        svm_preds["true_label"], svm_preds["predicted_label"],
        "Confusion Matrix: TF-IDF + SVM", "confusion_matrix_svm.png"
    )
    plot_confusion_matrix(
        codebert_preds["true_label"], codebert_preds["predicted_label"],
        "Confusion Matrix: CodeBERT", "confusion_matrix_codebert.png"
    )
    plot_f1_comparison(svm_metrics, codebert_metrics)
    plot_roc_curves(svm_preds, codebert_preds)

    print("\nSummary:")
    create_summary_table(svm_metrics, codebert_metrics)

    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
