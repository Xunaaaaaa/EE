"""
Obfuscate test scripts with techniques adapted from Invoke-Obfuscation
(Bohannon, 2016): case, token (backtick + variable renaming), concat,
string (format reordering), whitespace/comments and Base64 encoding, plus the
MPSD mixed_malicious "embedded" set.

The six transformations are applied to the malicious test scripts AND to the
benign test scripts (false-positive control). The embedded level is
malicious-only. Scripts are only ever handled as text.

Outputs per-level CSVs to data/obfuscation/seed_<seed>/ for evaluation.
"""

import hashlib
import re
import random
import base64
import pandas as pd
from config import PROCESSED_DIR, OBF_DIR, MPSD_DIR, RANDOM_SEED, read_script


# ---------------------------------------------------------------------------
# Level 1: Token obfuscation
# ---------------------------------------------------------------------------

COMMON_COMMANDS = [
    "Write-Host", "Write-Output", "Invoke-Expression", "Invoke-WebRequest",
    "Invoke-Command", "Invoke-Mimikatz", "Invoke-Shellcode",
    "New-Object", "Add-Type", "Set-Variable", "Get-Process",
    "Start-Process", "Get-Content", "Set-Content", "Out-File",
    "Get-ChildItem", "Remove-Item", "Get-WmiObject", "Get-ItemProperty",
    "Start-Sleep", "Test-Path", "ConvertTo-SecureString",
    "Import-Module", "Export-Csv", "ForEach-Object", "Where-Object",
    "Select-Object", "Sort-Object", "Measure-Object",
    "Download", "DownloadString", "DownloadFile", "DownloadData",
    "IEX", "iex",
]

def insert_ticks(command, rng):
    if len(command) < 3:
        return command
    chars = list(command)
    n_ticks = rng.randint(1, max(1, len(chars) // 3))
    positions = rng.sample(range(1, len(chars)), min(n_ticks, len(chars) - 1))
    for pos in sorted(positions, reverse=True):
        if chars[pos].isalpha():
            chars.insert(pos, '`')
    return ''.join(chars)


def obfuscate_token(script, rng):
    result = script

    for cmd in COMMON_COMMANDS:
        pattern = re.compile(re.escape(cmd), re.IGNORECASE)
        matches = list(pattern.finditer(result))
        for match in reversed(matches):
            original = match.group()
            obfuscated = insert_ticks(original, rng)
            result = result[:match.start()] + obfuscated + result[match.end():]

    def rename_var(m):
        name = m.group(1)
        if name.lower() in ('_', 'true', 'false', 'null', 'env', 'args',
                            'input', 'pshome', 'home', 'host', 'pid',
                            'error', 'profile', 'psversiontable'):
            return m.group(0)
        scrambled = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(len(name) + 2))
        return f'${scrambled}'

    seen_vars = {}
    def consistent_rename(m):
        name = m.group(1)
        if name.lower() in ('_', 'true', 'false', 'null', 'env', 'args',
                            'input', 'pshome', 'home', 'host', 'pid',
                            'error', 'profile', 'psversiontable'):
            return m.group(0)
        if name not in seen_vars:
            seen_vars[name] = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(max(len(name), 3)))
        return f'${seen_vars[name]}'

    result = re.sub(r'\$([a-zA-Z_]\w*)', consistent_rename, result)

    return result


# ---------------------------------------------------------------------------
# Level 2: String obfuscation (format-string reordering)
# ---------------------------------------------------------------------------

def obfuscate_string_concat(script, rng):
    result = re.sub(
        r"'([^']{4,})'",
        lambda m: _format_reorder(m.group(1), rng) if rng.random() > 0.3 else m.group(0),
        script
    )
    return result

def _format_reorder(content, rng):
    if len(content) < 4:
        return f"'{content}'"
    n_splits = rng.randint(1, min(3, len(content) // 2))
    positions = sorted(rng.sample(range(1, len(content)), min(n_splits, len(content) - 1)))

    parts = []
    prev = 0
    for pos in positions:
        parts.append(content[prev:pos])
        prev = pos
    parts.append(content[prev:])

    indices = list(range(len(parts)))
    shuffled = indices[:]
    rng.shuffle(shuffled)
    reverse_map = [0] * len(parts)
    for new_pos, orig_idx in enumerate(shuffled):
        reverse_map[orig_idx] = new_pos

    format_str = ''.join(f'{{{reverse_map[i]}}}' for i in range(len(parts)))
    args = ','.join(f"'{parts[shuffled[i]]}'" for i in range(len(parts)))
    return f'("{format_str}" -f {args})'


# ---------------------------------------------------------------------------
# Level 3: Encoding (Base64 launcher)
# ---------------------------------------------------------------------------

def obfuscate_encoding(script, rng):
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    launcher = f'powershell -nop -enc {encoded}'
    return launcher


# ---------------------------------------------------------------------------
# Level 4: Case randomization
# ---------------------------------------------------------------------------

def obfuscate_case(script, rng):
    result = []
    for char in script:
        if char.isalpha():
            result.append(char.upper() if rng.random() > 0.5 else char.lower())
        else:
            result.append(char)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Level 5: String concatenation (split commands into joined parts)
# ---------------------------------------------------------------------------

def obfuscate_concat(script, rng):
    result = script
    for cmd in COMMON_COMMANDS:
        pattern = re.compile(re.escape(cmd), re.IGNORECASE)
        matches = list(pattern.finditer(result))
        for match in reversed(matches):
            original = match.group()
            if len(original) < 4:
                continue
            n_splits = rng.randint(1, min(3, len(original) // 2))
            positions = sorted(rng.sample(range(1, len(original)), n_splits))
            parts = []
            prev = 0
            for pos in positions:
                parts.append(original[prev:pos])
                prev = pos
            parts.append(original[prev:])
            concatenated = '+'.join(f"'{p}'" for p in parts)
            # The call operator (&) makes the joined string run as a command;
            # a bare ('Inv'+'oke-Expression') would only evaluate to a string.
            result = result[:match.start()] + f"&({concatenated})" + result[match.end():]
    return result


# ---------------------------------------------------------------------------
# Level 6: Whitespace and comment injection
# ---------------------------------------------------------------------------

DECOY_COMMENTS = [
    "# initialization", "# process data", "# helper function",
    "# configuration", "# setup", "# cleanup", "# validation",
    "# check status", "# update values", "# temporary fix",
    "# logging", "# error handling", "# main logic",
    "# parse input", "# format output", "# load config",
]

def obfuscate_whitespace(script, rng):
    lines = script.split('\n')
    result = []
    for line in lines:
        if rng.random() > 0.5:
            result.extend([''] * rng.randint(1, 3))
        if rng.random() > 0.6:
            result.append(rng.choice(DECOY_COMMENTS))
        result.append(line)
    return '\n'.join(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEVELS = {
    "case": obfuscate_case,
    "token": obfuscate_token,
    "concat": obfuscate_concat,
    "string": obfuscate_string_concat,
    "whitespace": obfuscate_whitespace,
    "encoding": obfuscate_encoding,
}


def level_rng(level, cls):
    """Independent, reproducible RNG per (seed, level, class)."""
    digest = hashlib.sha256(f"{RANDOM_SEED}:{level}:{cls}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def obfuscate_set(df, level_name, obf_func, cls):
    rng = level_rng(level_name, cls)
    obfuscated_texts = []
    success_count = 0
    for script in df["script_text"]:
        try:
            obfuscated_texts.append(obf_func(script, rng))
            success_count += 1
        except Exception:
            obfuscated_texts.append(script)
    out_df = df[["source_file", "text_sha256", "label"]].copy()
    out_df.insert(0, "script_text", obfuscated_texts)
    return out_df, success_count


def build_mixed_set(malicious_test):
    """Embedded level: MPSD mixed_malicious files whose malicious component is a
    TEST-set script (so the payload was never seen in training), one per unique
    malicious test script. Filenames are <malicious_stem>_<benign_id>.ps1."""
    mixed_dir = MPSD_DIR / "mixed_malicious"
    pure_dir = MPSD_DIR / "malicious_pure"
    test_keys = set(malicious_test["text_sha256"])
    candidates = {}
    for f in sorted(mixed_dir.glob("*.ps1")):
        mal_stem, benign_id = f.stem.rsplit("_", 1)
        mal_text = read_script(pure_dir / f"{mal_stem}.ps1")
        key = hashlib.sha256(mal_text.strip().encode("utf-8")).hexdigest()
        if key in test_keys:
            candidates.setdefault(key, []).append((f, mal_stem, benign_id))
    rng = level_rng("mixed", "malicious")
    rows = []
    for key in sorted(candidates):
        f, mal_stem, benign_id = rng.choice(candidates[key])
        rows.append({
            "script_text": read_script(f),
            "source_file": f.name,
            "text_sha256": key,  # key of the embedded malicious test script
            "label": 1,
            "malicious_source_file": f"{mal_stem}.ps1",
            "benign_host_file": f"{benign_id}.ps1",
        })
    return pd.DataFrame(rows), len(candidates)


def main():
    OBF_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_df["script_text"] = test_df["script_text"].fillna("")
    malicious = test_df[test_df["label"] == 1].copy()
    benign = test_df[test_df["label"] == 0].copy()
    print(f"  Found {len(malicious)} malicious and {len(benign)} benign test scripts")

    for level_name, obf_func in LEVELS.items():
        print(f"\nApplying {level_name} obfuscation...")
        for cls, df, prefix in [("malicious", malicious, "obfuscated"), ("benign", benign, "benign_obfuscated")]:
            out_df, success_count = obfuscate_set(df, level_name, obf_func, cls)
            out_path = OBF_DIR / f"{prefix}_{level_name}.csv"
            out_df.to_csv(out_path, index=False)
            print(f"  {cls}: obfuscated {success_count}/{len(df)} scripts -> {out_path.name}")

    cols = ["script_text", "source_file", "text_sha256", "label"]
    malicious[cols].to_csv(OBF_DIR / "clean_malicious.csv", index=False)
    benign[cols].to_csv(OBF_DIR / "clean_benign.csv", index=False)
    print(f"\nSaved clean baselines to {OBF_DIR}")

    if (MPSD_DIR / "mixed_malicious").exists():
        print("\nBuilding embedded set from mixed_malicious (payload = test-set script)...")
        mixed_df, n_covered = build_mixed_set(malicious)
        mixed_df.to_csv(OBF_DIR / "obfuscated_mixed.csv", index=False)
        print(f"  {len(mixed_df)} embedded scripts ({n_covered}/{len(malicious)} malicious test scripts "
              f"have a mixed_malicious counterpart)")

    print("Done!")


if __name__ == "__main__":
    main()
