import hashlib
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
    # Sorted order + a stable (hashlib) suffix keep file names identical on
    # every machine; Python's built-in hash() is randomised per process.
    for src_name, dest_dir in [("powershell_benign_dataset", benign_dir), ("malicious_pure", malicious_dir)]:
        for ps1 in sorted((clone_dir / src_name).rglob("*.ps1")):
            dest = dest_dir / ps1.name
            if dest.exists():
                rel = ps1.relative_to(clone_dir).as_posix()
                dest = dest.with_stem(f"{dest.stem}_{hashlib.md5(rel.encode()).hexdigest()[:8]}")
            shutil.copy2(ps1, dest)

    benign_count = len(list(benign_dir.rglob("*.ps1")))
    malicious_count = len(list(malicious_dir.rglob("*.ps1")))
    print(f"Done: {benign_count} benign, {malicious_count} malicious scripts")
    print(f"(mixed_malicious available at {clone_dir / 'mixed_malicious'} for later obfuscation testing)")


if __name__ == "__main__":
    download_corpus()
