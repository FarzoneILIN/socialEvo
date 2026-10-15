import json
from pathlib import Path

D = Path("data/task3_fishing/batch")
files = sorted(D.glob("group_*.json"))
print(f"groups={len(files)}")

sigs = []
fb = 0
for f in files:
    data = json.loads(f.read_text(encoding="utf-8"))
    rp = len(data["rounds"])
    a = sum(r["decisions"]["A"]["value"] for r in data["rounds"])
    b = sum(r["decisions"]["B"]["value"] for r in data["rounds"])
    fb_a = sum(1 for r in data["rounds"] if not r["decisions"]["A"]["valid"])
    fb_b = sum(1 for r in data["rounds"] if not r["decisions"]["B"]["valid"])
    fb += fb_a + fb_b
    sig = (rp, a, b)
    sigs.append(sig)
    print(f"{f.name}: rounds={rp}, A_total={a}, B_total={b}, "
          f"fallback_A={fb_a}, fallback_B={fb_b}, nonce={data.get('nonce')}")

print(f"total_fallback_decisions={fb}")
print(f"unique_signatures={len(set(sigs))}/{len(sigs)}")
dups = sorted({s for s in set(sigs) if sigs.count(s) > 1})
print(f"duplicate_signatures={dups}")
