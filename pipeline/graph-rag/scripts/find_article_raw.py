
import sys
import re


def main():
    if len(sys.argv) < 3:
        print("استفاده: python find_article_raw.py <html_path> <article_number>")
        return

    html_path, number = sys.argv[1], sys.argv[2]

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # جستجوی "ماده 16" که بعدش رقم دیگه‌ای نیاد (که با "ماده 160" اشتباه نشه)
    pattern = re.compile(rf"ماده\s*{number}(?!\d)")
    matches = list(pattern.finditer(html))

    print(f"📏 حجم کل فایل: {len(html)} کاراکتر")
    print(f"🔎 تعداد رخداد «ماده {number}» در کل فایل: {len(matches)}")

    if not matches:
        print("❌ اصلاً پیدا نشد — یا شماره‌گذاری متفاوت است، یا این ماده اصلاً در فایل ذخیره‌شده وجود ندارد.")
        return

    for i, m in enumerate(matches):
        start = max(0, m.start() - 150)
        end = min(len(html), m.end() + 300)
        print(f"\n── رخداد {i + 1} از {len(matches)} (موقعیت کاراکتر {m.start()}) ──")
        print(html[start:end])
        print("...")


if __name__ == "__main__":
    main()