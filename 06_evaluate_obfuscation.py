"""
Evaluate both models (TF-IDF+SVM and CodeBERT) on the obfuscated test sets
produced by 05_obfuscate.py.

  - Malicious sets: detection rate (recall for the malicious class) per technique.
  - Benign sets (false-positive control): % wrongly flagged malicious per technique.
  - Embedded (mixed) set: where the malicious payload starts in CodeBERT tokens,
    and detection split into payload-visible vs payload-truncated scripts.
  - Token lengths of the clean malicious test scripts.

Generates (in results/seed_<seed>/):
  - obfuscation_predictions/<class>_<level>.csv  (per-script predictions)
  - obfuscation_metrics.csv       (malicious detection rate per level per model)
  - benign_fp_metrics.csv         (benign false-positive rate per level per model)
  - obfuscation_summary.json      (all of the above + payload and token stats)
  - obfuscation_detection_rate.png, benign_false_positive_rate.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.utils import logging as hf_logging
from config import MODELS_DIR, RESULTS_DIR, OBF_DIR, MPSD_DIR, RANDOM_SEED, read_script

MALICIOUS_LEVELS = ["clean", "case", "token", "concat", "string", "whitespace", "encoding", "mixed"]
BENIGN_LEVELS = ["clean", "case", "token", "concat", "string", "whitespace", "encoding"]
LEVEL_LABELS = {
    "clean": "Clean", "case": "Case\n(Randomize)", "token": "Token\n(Backtick)",
    "concat": "Concat\n(Split+Join)", "string": "String\n(Reorder)",
    "whitespace": "Whitespace\n(Comments)", "encoding": "Encoding\n(Base64)",
    "mixed": "Mixed\n(Embedded)",
}

CODEBERT_MODEL_DIR = RESULTS_DIR / "codebert_model"
CODEBERT_MAX_LENGTH = 512
# <s> and </s> take two of the 512 positions, leaving 510 content tokens.
CODEBERT_CONTENT_TOKENS = CODEBERT_MAX_LENGTH - 2
CODEBERT_BATCH_SIZE = 16
PRED_DIR = RESULTS_DIR / "obfuscation_predictions"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_svm():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    return vectorizer, model


def predict_svm(vectorizer, model, texts):
    X = vectorizer.transform(texts)
    return model.predict(X), model.decision_function(X)


def load_codebert(device):
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(CODEBERT_MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_codebert(tokenizer, model, texts, device):
    all_preds, all_probs = [], []
    for i in range(0, len(texts), CODEBERT_BATCH_SIZE):
        batch_texts = texts[i:i + CODEBERT_BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=CODEBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def n_tokens(tokenizer, text):
    """Untruncated CodeBERT length including <s> and </s>."""
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def load_level(cls, level):
    if level == "clean":
        path = OBF_DIR / f"clean_{cls}.csv"
    elif cls == "malicious":
        path = OBF_DIR / f"obfuscated_{level}.csv"
    else:
        path = OBF_DIR / f"benign_obfuscated_{level}.csv"
    df = pd.read_csv(path)
    df["script_text"] = df["script_text"].fillna("")
    return df


def evaluate_level(cls, level, svm_vec, svm_model, cb_tokenizer, cb_model, device):
    df = load_level(cls, level)
    texts = df["script_text"].tolist()
    svm_preds, svm_scores = predict_svm(svm_vec, svm_model, texts)
    cb_preds, cb_probs = predict_codebert(cb_tokenizer, cb_model, texts, device)
    lengths = [n_tokens(cb_tokenizer, t) for t in texts]

    out = df.drop(columns=["script_text"]).rename(columns={"label": "true_label"})
    out.insert(0, "level", level)
    out["svm_pred"] = svm_preds
    out["svm_decision_score"] = svm_scores
    out["codebert_pred"] = cb_preds
    out["codebert_malicious_prob"] = cb_probs
    out["n_tokens"] = lengths
    out["exceeds_512_tokens"] = out["n_tokens"] > CODEBERT_MAX_LENGTH
    return out, texts


def payload_analysis(pred_df, texts, tokenizer):
    """Locate the embedded malicious component and its first CodeBERT token."""
    char_starts, start_tokens, payload_lens = [], [], []
    for row, text in zip(pred_df.itertuples(), texts):
        payload = read_script(MPSD_DIR / "malicious_pure" / row.malicious_source_file).strip()
        start = text.find(payload)
        if start < 0:
            raise ValueError(f"payload not found in {row.source_file}")
        char_starts.append(start)
        start_tokens.append(len(tokenizer(text[:start], add_special_tokens=False)["input_ids"]))
        payload_lens.append(len(tokenizer(payload, add_special_tokens=False)["input_ids"]))

    pred_df["payload_char_start"] = char_starts
    # Index of the payload's first token among the content tokens (0-based);
    # its position in the model input is this + 1 because of <s>.
    pred_df["payload_start_token"] = start_tokens
    pred_df["payload_tokens"] = payload_lens
    pred_df["payload_visible"] = pred_df["payload_start_token"] < CODEBERT_CONTENT_TOKENS
    pred_df["payload_tokens_in_window"] = np.clip(
        CODEBERT_CONTENT_TOKENS - pred_df["payload_start_token"], 0, pred_df["payload_tokens"])

    def subset(mask):
        sub = pred_df[mask]
        return {
            "n_scripts": int(len(sub)),
            "codebert_detection_rate": float((sub["codebert_pred"] == 1).mean()) if len(sub) else None,
            "svm_detection_rate": float((sub["svm_pred"] == 1).mean()) if len(sub) else None,
        }

    visible = pred_df["payload_visible"]
    return {
        "definition": ("payload-visible = malicious component starts within CodeBERT's first "
                       f"{CODEBERT_CONTENT_TOKENS} content tokens (512-token window minus <s>, </s>); "
                       "payload-truncated = starts at/after that point"),
        "n_scripts": int(len(pred_df)),
        "payload_visible": subset(visible),
        "payload_truncated": subset(~visible),
        "pct_payload_truncated": float((~visible).mean() * 100),
        "pct_scripts_over_512_tokens_old_statistic": float(pred_df["exceeds_512_tokens"].mean() * 100),
        "median_payload_start_token": float(pred_df["payload_start_token"].median()),
    }


def plot_rates(df, levels, value_col, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(levels))
    for model, style, color, offset in [("TF-IDF + SVM", "o-", "#4C72B0", 12), ("CodeBERT", "s-", "#DD8452", -18)]:
        rates = df[df["model"] == model].set_index("level").loc[levels, value_col].values
        ax.plot(x, rates, style, color=color, linewidth=2, markersize=8, label=model)
        for i, r in enumerate(rates):
            ax.annotate(f"{r:.1%}", (x[i], r), textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=9, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[lv] for lv in levels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def main():
    hf_logging.set_verbosity_error()  # silence "sequence longer than 512" warnings when measuring lengths
    device = get_device()
    print(f"Using device: {device}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SVM model...")
    svm_vec, svm_model = load_svm()

    print("Loading CodeBERT model...")
    cb_tokenizer, cb_model = load_codebert(device)

    det_rows, fp_rows = [], []
    summary = {"seed": RANDOM_SEED, "device": str(device)}

    for cls, levels in [("malicious", MALICIOUS_LEVELS), ("benign", BENIGN_LEVELS)]:
        for level in levels:
            print(f"\n--- {cls} / {level} ---")
            pred_df, texts = evaluate_level(cls, level, svm_vec, svm_model, cb_tokenizer, cb_model, device)

            if cls == "malicious" and level == "mixed":
                summary["embedded_payload_analysis"] = payload_analysis(pred_df, texts, cb_tokenizer)
            if cls == "malicious" and level == "clean":
                summary["clean_malicious_token_lengths"] = {
                    "n_scripts": int(len(pred_df)),
                    "pct_exceeding_512_tokens": float(pred_df["exceeds_512_tokens"].mean() * 100),
                    "median_tokens": float(pred_df["n_tokens"].median()),
                    "definition": "CodeBERT tokens including <s> and </s>; >512 means the script is truncated",
                }
            if cls == "benign" and level == "clean":
                summary["clean_benign_token_lengths"] = {
                    "n_scripts": int(len(pred_df)),
                    "pct_exceeding_512_tokens": float(pred_df["exceeds_512_tokens"].mean() * 100),
                    "median_tokens": float(pred_df["n_tokens"].median()),
                }

            pred_df.to_csv(PRED_DIR / f"{cls}_{level}.csv", index=False)

            for model, col in [("TF-IDF + SVM", "svm_pred"), ("CodeBERT", "codebert_pred")]:
                flagged = int((pred_df[col] == 1).sum())
                rate = flagged / len(pred_df)
                row = {"model": model, "level": level, "n_scripts": len(pred_df), "n_flagged_malicious": flagged}
                if cls == "malicious":
                    row["detection_rate"] = rate
                    det_rows.append(row)
                else:
                    row["false_positive_rate"] = rate
                    fp_rows.append(row)
                print(f"  {model}: {'detection' if cls == 'malicious' else 'false-positive'} rate {rate:.4f} "
                      f"({flagged}/{len(pred_df)})")

    det_df = pd.DataFrame(det_rows)
    fp_df = pd.DataFrame(fp_rows)
    det_df.to_csv(RESULTS_DIR / "obfuscation_metrics.csv", index=False)
    fp_df.to_csv(RESULTS_DIR / "benign_fp_metrics.csv", index=False)
    summary["malicious_detection"] = det_rows
    summary["benign_false_positive"] = fp_rows
    with open(RESULTS_DIR / "obfuscation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved obfuscation_metrics.csv, benign_fp_metrics.csv, obfuscation_summary.json")

    print("\nGenerating charts...")
    plot_rates(det_df, MALICIOUS_LEVELS, "detection_rate", "Detection Rate (Recall for Malicious Class)",
               "Model Robustness Under Obfuscation (malicious test scripts)", "obfuscation_detection_rate.png")
    plot_rates(fp_df, BENIGN_LEVELS, "false_positive_rate", "False-Positive Rate (benign flagged malicious)",
               "False Positives Under Obfuscation (benign test scripts)", "benign_false_positive_rate.png")

    print("\n" + det_df.to_string(index=False))
    print("\n" + fp_df.to_string(index=False))
    print("\nEmbedded payload analysis:")
    print(json.dumps(summary["embedded_payload_analysis"], indent=2))
    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
