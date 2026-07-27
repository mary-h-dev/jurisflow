import json
from collections import Counter
d = json.load(open("data/features/civil/10035.json", encoding="utf-8"))
confs = [item["evidence"]["confidence"] for cat in ["roles","concepts","objects","actions","facts"] for item in d.get(cat,[])]
print(Counter(confs))