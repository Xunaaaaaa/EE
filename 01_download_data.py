import subprocess
import shutil
from pathlib import Path
from config import RAW_DIR

REPO_URL = "https://github.com/das-lab/mpsd.git"

def download_corpus():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "benign").exists() and (RAW_DIR / "malicious").exists():
        benign_count = len(list((RAW_DIR / "benign").rglob("*.ps1")))
        malicious_count = len(list((RAW_DIR / "malicious").rglob("*.ps1")))
        print(f"Data already exists: {benign_count} benign, {malicious_count} malicious")
        return

    clone_dir = RAW_DIR / "mpsd"
    if not clone_dir.exists():
        print(f"Cloning dataset from {REPO_URL}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
            check=True,
        )

    benign_dir = RAW_DIR / "benign"
    malicious_dir = RAW_DIR / "malicious"
    benign_dir.mkdir(exist_ok=True)
    malicious_dir.mkdir(exist_ok=True)

    print("Organizing scripts...")
    for ps1 in (clone_dir / "powershell_benign_dataset").rglob("*.ps1"):
        dest = benign_dir / ps1.name
        if dest.exists():
            dest = dest.with_stem(f"{dest.stem}_{hash(str(ps1)) % 100000}")
        shutil.copy2(ps1, dest)

    for ps1 in (clone_dir / "malicious_pure").rglob("*.ps1"):
        dest = malicious_dir / ps1.name
        if dest.exists():
            dest = dest.with_stem(f"{dest.stem}_{hash(str(ps1)) % 100000}")
        shutil.copy2(ps1, dest)

    benign_count = len(list(benign_dir.rglob("*.ps1")))
    malicious_count = len(list(malicious_dir.rglob("*.ps1")))
    print(f"Done: {benign_count} benign, {malicious_count} malicious scripts")
    print(f"(mixed_malicious available at {clone_dir / 'mixed_malicious'} for later obfuscation testing)")


if __name__ == "__main__":
    download_corpus()
