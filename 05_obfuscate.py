"""
Obfuscate malicious test scripts at three levels using techniques from
Invoke-Obfuscation (Bohannon, 2016):
  Level 1 - Token: backtick insertion in commands, variable renaming
  Level 2 - String: format-string reordering, string concatenation
  Level 3 - Encoding: Base64-encoded launcher wrapping

Outputs per-level CSVs to data/obfuscation/ for evaluation.
"""

import re
import random
import base64
import pandas as pd
from config import PROCESSED_DIR, DATA_DIR

OBF_DIR = DATA_DIR / "obfuscation"
RANDOM_SEED = 42


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

def insert_ticks(command):
    if len(command) < 3:
        return command
    chars = list(command)
    n_ticks = random.randint(1, max(1, len(chars) // 3))
    positions = random.sample(range(1, len(chars)), min(n_ticks, len(chars) - 1))
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
            obfuscated = insert_ticks(original)
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
            result = result[:match.start()] + f"({concatenated})" + result[match.end():]
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
# Level 7: Layered (token + string combined)
# ---------------------------------------------------------------------------

def obfuscate_layered(script, rng):
    result = obfuscate_token(script, rng)
    result = obfuscate_string_concat(result, rng)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEVELS = {
    "token": obfuscate_token,
    "string": obfuscate_string_concat,
    "case": obfuscate_case,
    "concat": obfuscate_concat,
    "whitespace": obfuscate_whitespace,
    "layered": obfuscate_layered,
    "encoding": obfuscate_encoding,
}


def main():
    rng = random.Random(RANDOM_SEED)
    OBF_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_df["script_text"] = test_df["script_text"].fillna("")
    malicious = test_df[test_df["label"] == 1].copy()
    print(f"  Found {len(malicious)} malicious test scripts")

    for level_name, obf_func in LEVELS.items():
        print(f"\nApplying {level_name} obfuscation...")
        obfuscated_texts = []
        success_count = 0

        for _, row in malicious.iterrows():
            script = row["script_text"]
            try:
                if level_name == "encoding":
                    obf = obf_func(script, rng)
                else:
                    obf = obf_func(script, rng)
                obfuscated_texts.append(obf)
                success_count += 1
            except Exception as e:
                obfuscated_texts.append(script)

        out_df = pd.DataFrame({
            "script_text": obfuscated_texts,
            "label": 1,
        })
        out_path = OBF_DIR / f"obfuscated_{level_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Obfuscated {success_count}/{len(malicious)} scripts")
        print(f"  Saved to {out_path}")

    malicious[["script_text", "label"]].to_csv(OBF_DIR / "clean_malicious.csv", index=False)
    print(f"\nSaved clean malicious baseline to {OBF_DIR / 'clean_malicious.csv'}")

    mixed_dir = DATA_DIR / "raw" / "mpsd" / "mixed_malicious"
    if mixed_dir.exists():
        print("\nLoading mixed_malicious scripts (malicious embedded in benign)...")
        from pathlib import Path
        ps1_files = sorted(mixed_dir.glob("*.ps1"))
        sampled = rng.sample(ps1_files, min(len(malicious), len(ps1_files)))
        mixed_texts = []
        for f in sampled:
            try:
                mixed_texts.append(f.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
        import csv
        mixed_df = pd.DataFrame({"script_text": mixed_texts, "label": 1})
        mixed_df.to_csv(OBF_DIR / "obfuscated_mixed.csv", index=False,
                        quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"  Sampled {len(mixed_texts)} mixed_malicious scripts")

    print("Done!")


if __name__ == "__main__":
    main()
