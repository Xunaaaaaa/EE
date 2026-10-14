import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MPSD_DIR = RAW_DIR / "mpsd"

# The seed controls the split, SVM, CodeBERT init/shuffling and obfuscation.
# Override with the EE_SEED environment variable to run extra seeds.
RANDOM_SEED = int(os.environ.get("EE_SEED", "42"))

# Every seed gets its own processed data, models and results so runs never mix.
PROCESSED_DIR = DATA_DIR / "processed" / f"seed_{RANDOM_SEED}"
OBF_DIR = DATA_DIR / "obfuscation" / f"seed_{RANDOM_SEED}"
MODELS_DIR = PROJECT_ROOT / "models" / f"seed_{RANDOM_SEED}"
RESULTS_DIR = PROJECT_ROOT / "results" / f"seed_{RANDOM_SEED}"

BENIGN_DOWNSAMPLE = None

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_ANALYZER = "char"
TFIDF_SUBLINEAR_TF = True
SVM_C_VALUES = [0.01, 0.1, 1, 10]

CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_EPOCHS = 4
CODEBERT_LR = 2e-5
CODEBERT_BATCH_SIZE = 16
CODEBERT_MAX_LENGTH = 512
CODEBERT_WARMUP_RATIO = 0.1
CODEBERT_WEIGHT_DECAY = 0.01


def read_script(path):
    """Read a .ps1 file as TEXT only (never executed).

    Uses utf-8-sig so a UTF-8 byte-order mark is removed: in MPSD ~35% of
    benign files start with a BOM and no malicious file does, which would give
    both models a trivial non-content cue. Stray NUL bytes (a handful of
    mis-encoded UTF-16 files) are dropped for the same reason.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return text.replace("\x00", "")
