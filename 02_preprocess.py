import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import (
    RAW_DIR, PROCESSED_DIR, BENIGN_DOWNSAMPLE,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
)

def read_scripts(directory, label):
    records = []
    for ps1 in directory.rglob("*.ps1"):
        try:
            text = ps1.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) > 0:
                records.append({"script_text": text, "label": label})
        except Exception:
            continue
    return records


def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading benign scripts...")
    benign = read_scripts(RAW_DIR / "benign", label=0)
    print(f"  Found {len(benign)} benign scripts")

    print("Reading malicious scripts...")
    malicious = read_scripts(RAW_DIR / "malicious", label=1)
    print(f"  Found {len(malicious)} malicious scripts")

    df_benign = pd.DataFrame(benign)
    df_malicious = pd.DataFrame(malicious)

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

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\nSplits saved to {PROCESSED_DIR}:")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_benign = (split_df["label"] == 0).sum()
        n_malicious = (split_df["label"] == 1).sum()
        print(f"  {name}: {len(split_df)} total ({n_benign} benign, {n_malicious} malicious)")


if __name__ == "__main__":
    preprocess()
