from pathlib import Path
import json
import platform
import sys
import time

result = {
    "message": "Kaggle ran this script",
    "python": sys.version,
    "platform": platform.platform(),
    "timestamp": time.time(),
    "sum_1_to_10": sum(range(1, 11)),
}

Path("/kaggle/working/proof_result.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("DONE: wrote /kaggle/working/proof_result.json")
