"""
inspect_case.py — بررسی ساختار HTML یک صفحه‌ی رأی در ara.jri.ac.ir

به‌جای چاپ کل HTML خام (که پر از CSS/JS بی‌ربطه)، این اسکریپت:
    ۱) دنبال برچسب‌های متادیتا می‌گردد (شماره دادنامه، تاریخ، گروه، مرجع، ...)
       و تگ/کلاسِ دقیقِ اطراف هرکدام را نشان می‌دهد
    ۲) بزرگ‌ترین بلوک متنی صفحه را پیدا می‌کند — که تقریباً همیشه متن
       اصلیِ رأی است (چون از هر متن دیگری در صفحه طولانی‌تره)

اجرا:
    python inspect_case.py https://ara.jri.ac.ir/Judge/Text/32208
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

LABELS_TO_FIND = [
    "شماره پرونده",
    "شماره دادنامه",
    "تاریخ دادنامه",
    "گروه",
    "نوع رای",
    "نوع رأی",
    "مرجع رسیدگی",
    "مرجع صدور",
    "شعبه",
]


def main():
    if len(sys.argv) < 2:
        print("استفاده: python inspect_case.py <url>")
        return

    url = sys.argv[1]
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    print(f"📌 عنوان صفحه: {soup.title.get_text() if soup.title else '-'}")
    print(f"📏 حجم HTML: {len(r.text)} کاراکتر\n")

    print("── جستجوی برچسب‌های متادیتا ──")
    seen = set()
    for label in LABELS_TO_FIND:
        if label in seen:
            continue
        text_nodes = soup.find_all(string=lambda t: t and label in t)
        for t in text_nodes[:1]:
            seen.add(label)
            parent = t.parent
            container = parent.parent if parent.parent else parent
            print(f"\n🏷️  «{label}»")
            print(f"   تگ مستقیم: <{parent.name} class=\"{parent.get('class')}\" id=\"{parent.get('id')}\">")
            print(f"   HTML اطراف (۴۰۰ کاراکتر): {str(container)[:400]}")

    print("\n\n── بزرگ‌ترین بلوک متنی صفحه (احتمالاً متن اصلی رأی) ──")
    candidates = soup.find_all(["p", "div"])
    candidates_sorted = sorted(
        candidates, key=lambda el: len(el.get_text(strip=True)), reverse=True
    )
    if candidates_sorted:
        biggest = candidates_sorted[0]
        print(f"تگ: <{biggest.name} class=\"{biggest.get('class')}\" id=\"{biggest.get('id')}\">")
        print(f"طول متن: {len(biggest.get_text(strip=True))} کاراکتر")
        print(f"نمونه‌ی ۴۰۰ کاراکتر اول:\n{biggest.get_text(strip=True)[:400]}")


if __name__ == "__main__":
    main()