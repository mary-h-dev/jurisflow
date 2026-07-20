"""
diagnose_gaps.py — پیدا کردن ماده‌های گم‌شده در یک قانون parse‌شده

این اسکریپت:
    ۱) از روی JSON خروجی، شماره‌ی مواد موجود را می‌خواند
    ۲) شکاف‌های بین شماره‌ها را پیدا می‌کند (مثلاً اگر 45 هست ولی 46 نیست)
    ۳) برای چند تا از این شکاف‌ها، HTML خام اطرافش را از فایل اصلی
       استخراج می‌کند تا ببینیم چرا آن ماده پارس نشده

اجرا:
    python diagnose_gaps.py data/laws/civil_procedure_law.json data/laws/civil_procedure_law.html
"""

import sys
import json
import re
from bs4 import BeautifulSoup


def find_gaps(article_numbers: list[int]) -> list[int]:
    """شماره ماده‌هایی که بین کمترین و بیشترین شماره‌ی موجود، جا افتاده‌اند"""
    if not article_numbers:
        return []
    full_range = set(range(min(article_numbers), max(article_numbers) + 1))
    return sorted(full_range - set(article_numbers))


def main():
    if len(sys.argv) < 3:
        print("استفاده: python diagnose_gaps.py <json_path> <html_path>")
        return

    json_path, html_path = sys.argv[1], sys.argv[2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article_numbers = [a["article_number"] for a in data["articles"]]
    print(f"📊 تعداد کل مواد parse‌شده: {len(article_numbers)}")
    print(f"📊 بازه: از ماده {min(article_numbers)} تا ماده {max(article_numbers)}")

    gaps = find_gaps(article_numbers)
    print(f"\n🕳️  تعداد شکاف (ماده‌های گم‌شده در این بازه): {len(gaps)}")
    if gaps:
        print(f"   شماره‌ها: {gaps[:40]}{' ...' if len(gaps) > 40 else ''}")

    if not gaps:
        print("✅ هیچ شکافی نیست — همه‌چیز پشت سر هم است.")
        return

    # حالا برای چند تا از گم‌شده‌ها، متن خام اطرافش رو از HTML اصلی پیدا کن
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    all_text_elements = soup.find_all(["p", "div", "span"])
    print(f"\n🔬 بررسی HTML خام برای {min(3, len(gaps))} نمونه از ماده‌های گم‌شده:\n")

    for missing_num in gaps[:3]:
        pattern = re.compile(rf"ماده\s*{missing_num}\b")
        found_els = [el for el in all_text_elements if pattern.search(el.get_text())]

        print(f"── ماده {missing_num} ──")
        if not found_els:
            print("   ❌ اصلاً در HTML پیدا نشد (شاید شماره‌گذاری با حروف/الگوی متفاوت است)")
        else:
            for el in found_els[:1]:
                print(f"   تگ: <{el.name} class=\"{el.get('class')}\">")
                print(f"   متن (۲۰۰ کاراکتر اول): {el.get_text(strip=True)[:200]}")
        print()


if __name__ == "__main__":
    main()