"""
count_cases.py — شمارش تعداد رأی‌های مرتبط با هر یک از ۵ قانون مادر
در سامانه ملی آراء (ara.jri.ac.ir)، بدون دانلود متن کامل هیچ رأی‌ای.

فقط صفحه‌ی اول نتایج هر قانون را می‌گیرد (یک درخواست سبک) و عدد
«تعداد یافته‌ها» را از آن استخراج می‌کند — هیچ اتصالی به Neo4j ندارد،
هیچ داده‌ای ذخیره نمی‌کند، فقط یک گزارش عددی چاپ می‌کند.

اجرا:
    python count_cases.py
"""

import re
import time
import requests


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

# کلید → (نام نمایشی، شناسه‌ی Laws در سامانه)
LAW_IDS = {
    "civil":               ("قانون مدنی", 2),
    "civil_procedure":     ("آیین دادرسی مدنی", 4671),
    "commercial":          ("قانون تجارت", 2970),
    "penal":               ("قانون مجازات اسلامی", 867),
    "criminal_procedure":  ("آیین دادرسی کیفری", 2446),
}

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# الگوی جدید و منعطف‌تر:
# این الگو تگ‌های b، فاصله‌ها، نیم‌فاصله‌ها و کاراکترهای تمیزکاری شده را پوشش می‌دهد.
COUNT_PATTERN = re.compile(
    r"تعداد[\s\u200c]*یافته[\s\u200c]*ها(?:</b>)?[\s\u200c]*:?[\s\u200c]*([0-9۰-۹]+)",
    re.IGNORECASE
)


def count_for(law_id: int) -> int | None:
    url = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"

    # حذف تگ‌های b اضافه برای راحت‌تر شدن کار Regex
    clean_text = response.text.replace("<b>", "").replace("</b>", "")

    match = COUNT_PATTERN.search(clean_text)
    if not match:
        return None

    raw_number = match.group(1).translate(PERSIAN_DIGITS)
    return int(raw_number)


def main():
    print("🔎 در حال شمارش تعداد آراء هر قانون (بدون دانلود متن)...\n")

    total = 0
    results = {}

    for key, (name, law_id) in LAW_IDS.items():
        count = count_for(law_id)
        if count is None:
            print(f"⚠️  {name:35s} → عدد پیدا نشد (باید دستی بررسی شود)")
        else:
            print(f"📊 {name:35s} → {count:,} رأی")
            results[key] = count
            total += count

        time.sleep(2)  # احترام به نرخ درخواست سایت

    print(f"\n📈 مجموع کل (هر ۵ قانون): {total:,} رأی")


if __name__ == "__main__":
    main()