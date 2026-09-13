"""
features/pilot_stats.py — overall stats over pilot outputs, without
needing to manually read each file one by one.
uv run -m features.pilot_stats
"""
import glob
import json
from collections import Counter

def run(features_dir="data/features/civil"):
    paths = glob.glob(f"{features_dir}/*.json")
    print(f"📂 {len(paths)} cases")

    total_features = 0
    null_count = 0
    confidences = []
    role_counts = Counter()
    files_missing_khahan_khoonde = []

    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        role_values = {r["value"] for r in d.get("roles", [])}
        if "خواهان" not in role_values and "خوانده" not in role_values:
            files_missing_khahan_khoonde.append(d["ruling_id"])

        for cat in ["roles", "concepts", "objects", "actions", "facts"]:
            for item in d.get(cat, []):
                total_features += 1
                ev = item.get("evidence", {})
                if ev.get("start_char") is None:
                    null_count += 1
                confidences.append(ev.get("confidence"))
                if cat == "roles":
                    role_counts[item["value"]] += 1

    print(f"📊 Total Features: {total_features}")
    print(f"🔴 Null rate (evidence location failed): {null_count}/{total_features} ({null_count/total_features:.1%})")
    print(f"📈 Confidence distribution: {Counter(confidences)}")
    print(f"⚠️ Cases missing plaintiff/defendant (خواهان/خوانده): {len(files_missing_khahan_khoonde)} → {files_missing_khahan_khoonde[:10]}")
    print(f"🏷️ Most common roles: {role_counts.most_common(15)}")

if __name__ == "__main__":
    run()