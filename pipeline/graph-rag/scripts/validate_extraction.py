"""
validate_extraction.py — اعتبارسنجی خودکار استخراج مواد از متن آزاد

کارش: توی همه‌ی فایل‌های JSON اسکرپ‌شده می‌گرده، برای هر پرونده چک می‌کنه:
    آیا توی متن پرونده کلمه‌ی «ماده» یا «مواد» اومده، ولی
    text_cited_articles هنوز خالی مونده؟
اگه بله، یعنی جایی رو از قلم انداختیم (یا قانونی خارج از ۵ قانون مادرمون
ذکر شده، که طبیعیه، یا واقعاً یک باگ دیگه‌ست) — این پرونده‌ها رو لیست
می‌کنه تا دستی بررسی بشن.

اجرا:
    python validate_extraction.py data/cases/penal
    python validate_extraction.py data/cases          ← همه‌ی قوانین با هم
"""

import sys
import os
import json
import re


ARTICLE_WORD_PATTERN = re.compile(r"ماده|مواد")


def check_file(path: str) -> dict | None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_text = " ".join(s.get("text", "") for s in data.get("sections", []))
    has_article_word = bool(ARTICLE_WORD_PATTERN.search(full_text))
    text_cited = data.get("text_cited_articles", [])
    official_cited = data.get("cited_articles", [])

    if has_article_word and not text_cited and not official_cited:
        return {
            "ruling_id": data.get("ruling_id"),
            "title": data.get("title", "")[:60],
            "text_length": len(full_text),
            "path": path,
        }
    return None


def walk_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                yield os.path.join(dirpath, fn)


def main():
    if len(sys.argv) < 2:
        print("استفاده: python validate_extraction.py <data/cases یا data/cases/<law_key>>")
        return

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"❌ پوشه پیدا نشد: {root}")
        return

    total = 0
    suspicious = []

    for path in walk_json_files(root):
        total += 1
        result = check_file(path)
        if result:
            suspicious.append(result)

    print(f"📊 کل پرونده‌های بررسی‌شده: {total}")
    print(f"⚠️  پرونده‌های مشکوک (کلمه‌ی ماده/مواد داره ولی هیچ ماده‌ای استخراج نشده): {len(suspicious)}\n")

    for s in suspicious:
        print(f"  #{s['ruling_id']:8s}  ({s['text_length']:6d} کاراکتر متن)  {s['title']}")
        print(f"           فایل: {s['path']}")

    if total > 0:
        rate = 100 * (total - len(suspicious)) / total
        print(f"\n✅ نرخ موفقیت استخراج (حداقل یک ماده پیدا شده یا رسمی یا متنی): {rate:.1f}%")


if __name__ == "__main__":
    main()