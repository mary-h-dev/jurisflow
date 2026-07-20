"""
inspect_listing.py — بررسی کامل ساختار صفحه‌ی لیست و صفحه‌بندی (بدون فیلتر)

اجرا:
    python inspect_listing.py 867
"""

import sys
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}


def main():
    law_id = sys.argv[1]
    url = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}"

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    print(f"📏 حجم HTML: {len(r.text)} کاراکتر\n")

    # همه‌ی لینک‌های صفحه، بدون فیلتر
    all_links = soup.find_all("a")
    print(f"🔗 تعداد کل تگ‌های <a> در صفحه: {len(all_links)}\n")
    for a in all_links:
        href = a.get("href", "")
        text = a.get_text(strip=True)
        # فقط اونایی که به /Judge/Text/ نمیرن رو نشون بده (چون اونا رو قبلا دیدیم)
        if "/Judge/Text/" not in href and text:
            print(f"   href=«{href}»   متن=«{text[:40]}»   onclick=«{a.get('onclick')}»")

    # همه‌ی input های مخفی، بدون محدودیت تعداد
    print("\n🔎 همه‌ی input های مخفی:")
    hidden_inputs = soup.find_all("input", type="hidden")
    for inp in hidden_inputs:
        print(f"   name={inp.get('name')}  value={inp.get('value')}")

    # همه‌ی فرم‌ها و action‌شون
    print("\n📋 فرم‌های صفحه:")
    for form in soup.find_all("form"):
        print(f"   action={form.get('action')}  method={form.get('method')}  id={form.get('id')}")

    # دنبال هر چیزی با متن فارسی مربوط به صفحه‌بندی
    print("\n🔎 عناصر حاوی کلمات صفحه‌بندی (بعدی/قبلی/صفحه):")
    for el in soup.find_all(string=lambda t: t and any(
        w in t for w in ["بعدی", "قبلی", "صفحه ۲", "صفحه بعد", "Next", "»"]
    )):
        parent = el.parent
        print(f"   متن=«{el.strip()[:30]}»  تگ والد=<{parent.name} class=\"{parent.get('class')}\">")


if __name__ == "__main__":
    main()