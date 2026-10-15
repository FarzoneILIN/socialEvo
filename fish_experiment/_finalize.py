"""后台收尾脚本：等待批量实验跑完后自动生成分析报告。

由外部后台启动（非交互），最多等 60 分钟：
    等待 fish_experiment_batch.json 出现 -> 运行 analyze_batch.main() -> 写完成标记。
"""
import time
from pathlib import Path

from code.task3_fishing import analyze_batch

AGG = Path("data/task3_fishing/fish_experiment_batch.json")
MARK = Path("data/task3_fishing/_BATCH_REPORT_DONE")

for _ in range(240):  # 最多 60 分钟
    if AGG.exists():
        try:
            analyze_batch.main()
            MARK.write_text("done", encoding="utf-8")
            print("REPORT DONE", flush=True)
        except Exception as e:  # 生成报告失败也不影响主实验
            print(f"REPORT FAILED: {e}", flush=True)
            MARK.write_text(f"failed: {e}", encoding="utf-8")
        break
    time.sleep(15)
else:
    print("TIMEOUT waiting for batch", flush=True)
