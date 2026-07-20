"""
debug_check.py — بررسی سریع اینکه HTML ذخیره‌شده واقعاً محتوا دارد یا نه
اجرا کن: python debug_check.py data/laws/civil_law.html
"""
import sys
from bs4 import BeautifulSoup

path = sys.argv[1] if len(sys.argv) > 1 else "data/laws/civil_law.html"

with open(path, "r", encoding="utf-8") as f:
    html = f.read()

print(f"📏 حجم فایل: {len(html)} کاراکتر")

soup = BeautifulSoup(html, "lxml")

# چک کن آیا اصلاً p.SecTex پیدا میشه
elements = soup.find_all("p", class_="SecTex")
print(f"🔍 تعداد p.SecTex پیدا شده: {len(elements)}")

if len(elements) == 0:
    # شاید کلاس دیگه‌ای استفاده شده — همه‌ی کلاس‌های p رو نشون بده
    all_p = soup.find_all("p")
    print(f"📄 تعداد کل تگ‌های p: {len(all_p)}")
    classes_found = set()
    for p in all_p[:50]:
        if p.get("class"):
            classes_found.add(" ".join(p.get("class")))
    print(f"🏷️  کلاس‌های p موجود (نمونه): {classes_found}")

    # چک کن آیا صفحه کپچا یا خطا برگردونده
    title = soup.find("title")
    print(f"📌 عنوان صفحه: {title.get_text() if title else 'یافت نشد'}")
    print(f"📝 نمونه‌ی ۵۰۰ کاراکتر اول متن صفحه:")
    print(soup.get_text()[:500])