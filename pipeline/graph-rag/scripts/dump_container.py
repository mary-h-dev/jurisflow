"""
dump_container.py — نمایش HTML خام داخل div.container

هدف: پیدا کردن این‌که متن اصلی رأی (بعد از دکمه‌های منو مثل «چاپ متن»)
داخل چه زیرتگی قرار داره — تا بتونیم دقیقاً همون رو انتخاب کنیم و
دکمه‌های بی‌ربط رو کنار بذاریم.

اجرا:
    python dump_container.py https://ara.jri.ac.ir/Judge/Text/32208
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
    url = sys.argv[1]
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    containers = soup.find_all("div", class_="container")
    print(f"📦 تعداد div.container در صفحه: {len(containers)}\n")

    for i, container in enumerate(containers):
        html = str(container)
        print(f"── container شماره {i} (طول: {len(html)} کاراکتر) ──")
        print(html[:3000])
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()