import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_DIR, PROCESSED_DIR, RESULTS_DIR, BENIGN_DOWNSAMPLE,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, read_script
)


def text_key(text):
    """Deduplication key: SHA-256 of the stripped script text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def read_scripts(directory, label):
    records = []
    n_empty = 0
    # Sorted so the dataset (and therefore the split) is identical on any machine.
    for ps1 in sorted(directory.rglob("*.ps1")):
        try:
            text = read_script(ps1)
        except Exception:
            continue
        if len(text.strip()) == 0:
            n_empty += 1
            continue
        records.append({
            "script_text": text,
            "label": label,
            "source_file": ps1.name,
            "text_sha256": text_key(text),
        })
    return pd.DataFrame(records), n_empty


def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading benign scripts...")
    df_benign, empty_benign = read_scripts(RAW_DIR / "benign", label=0)
    print(f"  Found {len(df_benign)} non-empty benign files ({empty_benign} empty skipped)")

    print("Reading malicious scripts...")
    df_malicious, empty_malicious = read_scripts(RAW_DIR / "malicious", label=1)
    print(f"  Found {len(df_malicious)} non-empty malicious files ({empty_malicious} empty skipped)")

    # Texts labelled both benign and malicious are ambiguous: drop every copy.
    both = set(df_benign["text_sha256"]) & set(df_malicious["text_sha256"])
    both_benign_files = int(df_benign["text_sha256"].isin(both).sum())
    both_malicious_files = int(df_malicious["text_sha256"].isin(both).sum())
    df_benign = df_benign[~df_benign["text_sha256"].isin(both)]
    df_malicious = df_malicious[~df_malicious["text_sha256"].isin(both)]
    print(f"\nRemoved {len(both)} scripts labelled BOTH benign and malicious "
          f"({both_benign_files} benign files, {both_malicious_files} malicious files)")

    # Keep one copy of each unique script (first in sorted path order).
    dup_benign = int(df_benign.duplicated("text_sha256").sum())
    dup_malicious = int(df_malicious.duplicated("text_sha256").sum())
    df_benign = df_benign.drop_duplicates("text_sha256", keep="first")
    df_malicious = df_malicious.drop_duplicates("text_sha256", keep="first")
    print(f"Dropped duplicates: {dup_benign} benign, {dup_malicious} malicious "
          f"({dup_benign + dup_malicious} total)")
    print(f"Unique scripts: {len(df_benign)} benign, {len(df_malicious)} malicious")

    if BENIGN_DOWNSAMPLE and len(df_benign) > BENIGN_DOWNSAMPLE:
        print(f"Downsampling benign from {len(df_benign)} to {BENIGN_DOWNSAMPLE}...")
        df_benign = df_benign.sample(n=BENIGN_DOWNSAMPLE, random_state=RANDOM_SEED)

    df = pd.concat([df_benign, df_malicious], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Total dataset: {len(df)} scripts ({(df['label'] == 0).sum()} benign, {(df['label'] == 1).sum()} malicious)")

    val_test_ratio = VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO)
    train_df, temp_df = train_test_split(
        df, test_size=val_test_ratio, random_state=RANDOM_SEED, stratify=df["label"]
    )

    relative_val = VAL_RATIO / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=RANDOM_SEED, stratify=temp_df["label"]
    )

    # Sanity check: no script text may appear in more than one split.
    keys = [set(d["text_sha256"]) for d in (train_df, val_df, test_df)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]), "split leakage"

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    stats = {
        "seed": RANDOM_SEED,
        "raw_files_benign": len(df_benign) + dup_benign + both_benign_files + empty_benign,
        "raw_files_malicious": len(df_malicious) + dup_malicious + both_malicious_files + empty_malicious,
        "empty_files_skipped": {"benign": empty_benign, "malicious": empty_malicious},
        "both_labelled_unique_scripts_removed": len(both),
        "both_labelled_files_removed": {"benign": both_benign_files, "malicious": both_malicious_files},
        "duplicates_dropped": {"benign": dup_benign, "malicious": dup_malicious,
                               "total": dup_benign + dup_malicious},
        "unique_benign": len(df_benign),
        "unique_malicious": len(df_malicious),
        "splits": {},
    }

    print(f"\nSplits saved to {PROCESSED_DIR}:")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_benign = int((split_df["label"] == 0).sum())
        n_malicious = int((split_df["label"] == 1).sum())
        stats["splits"][name] = {"total": len(split_df), "benign": n_benign, "malicious": n_malicious}
        print(f"  {name}: {len(split_df)} total ({n_benign} benign, {n_malicious} malicious)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (PROCESSED_DIR, RESULTS_DIR):
        with open(out_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    preprocess()
