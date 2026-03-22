#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
log = ROOT / "results" / "logs" / "run_cipher_ablations_extended_multiseed.log"
text = log.read_text(errors="ignore") if log.exists() else ""
pat = re.compile(r"CIPHER ablations \(8-8\):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^,]*,\s*([0-9.]+)s/run\]")
matches = pat.findall(text)
if not matches:
    print("No tqdm progress found yet")
    raise SystemExit(0)

pct_s, done_s, total_s, spr_s = matches[-1]
pct = int(pct_s)
done = int(done_s)
total = int(total_s)
spr = float(spr_s)
remaining = max(total - done, 0)
eta = int(remaining * spr)
bar_len = 30
filled = int(bar_len * done / total) if total else 0
bar = "#" * filled + "-" * (bar_len - filled)

print(f"Progress: [{bar}] {pct}% ({done}/{total})")
print(f"Speed: {spr:.1f} sec/run")
print(f"ETA: {eta//3600:02d}:{(eta%3600)//60:02d}:{eta%60:02d}")
