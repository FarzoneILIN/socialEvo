import time
from pathlib import Path

D = Path("data/task3_fishing/batch")
OUT = Path("data/task3_fishing/fish_experiment_batch.json")
LOG = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\cline\proceed-while-running-1792070364274-2dkfmq7.log")

last = -1
last_warn = -1
for _ in range(360):  # 最多 90 分钟
    n = len(list(D.glob("group_*.json"))) if D.exists() else 0
    done = OUT.exists()
    w = 0
    if LOG.exists():
        try:
            w = LOG.read_text(encoding="utf-8", errors="ignore").count("警告")
        except Exception:
            pass
    if n != last or w != last_warn or done:
        last, last_warn = n, w
        print(f"[{time.strftime('%H:%M:%S')}] groups={n}/10, fallback_warnings={w}, "
              f"aggregate={'done' if done else 'pending'}", flush=True)
    if done:
        print("BATCH COMPLETE", flush=True)
        break
    time.sleep(15)
else:
    print("TIMEOUT (90min) while waiting for batch", flush=True)
