
# DJANGO_SETTINGS_MODULE=config.settings.development python -c "import json; data = json.load(open('apps/search/calibration/data/annotations/case_grounded.json', encoding='utf-8')); articles = sorted({a for r in data for a in r.get('gold_articles', [])}); print(f'تعداد: {len(articles)}'); print('\n'.join(articles))"

import json
from pathlib import Path

file_path = Path("apps/search/calibration/data/annotations/case_grounded.json")

if not file_path.exists():
    print(f"Error: File not found at {file_path}")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_articles = set()
    for ruling in data:
        for article in ruling.get("gold_articles", []):
            unique_articles.add(article)

    print(f"تعداد کل مواد منحصربه‌فرد: {len(unique_articles)}\n" + "-" * 30)
    for a in sorted(unique_articles):
        print(a)