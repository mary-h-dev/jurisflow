"""
verify_pagination.py — تایید اینکه PageNumber/PageSize واقعاً روی GET کار می‌کنن

اجرا:
    python verify_pagination.py 867
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


def count_links(html: str):
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=lambda h: h and "/Judge/Text/" in h)
    return [a.get("href").split("/")[-1] for a in links]


def main():
    law_id = sys.argv[1]

    # تست ۱: PageSize بزرگ — باید همه‌ی نتایج رو یک‌جا بده
    url1 = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}&PageSize=2000"
    r1 = requests.get(url1, headers=HEADERS, timeout=15)
    r1.encoding = "utf-8"
    ids1 = count_links(r1.text)
    print(f"PageSize=2000 → {len(ids1)} لینک منحصربه‌فرد: {len(set(ids1))}")

    # تست ۲: صفحه‌ی دوم با PageSize پیش‌فرض — باید نتایج متفاوت بده
    url2 = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}&PageNumber=2"
    r2 = requests.get(url2, headers=HEADERS, timeout=15)
    r2.encoding = "utf-8"
    ids2 = count_links(r2.text)
    print(f"PageNumber=2 (پیش‌فرض PageSize=25) → {len(ids2)} لینک")
    print(f"   نمونه: {ids2[:3]}")

    # تست ۳: صفحه‌ی اول برای مقایسه
    url3 = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}"
    r3 = requests.get(url3, headers=HEADERS, timeout=15)
    r3.encoding = "utf-8"
    ids3 = count_links(r3.text)
    print(f"بدون پارامتر (صفحه ۱، پیش‌فرض) → {len(ids3)} لینک")
    print(f"   نمونه: {ids3[:3]}")

    print(f"\nآیا صفحه ۱ و ۲ متفاوتن؟ {'بله ✅' if set(ids2) != set(ids3) else 'نه ❌'}")


if __name__ == "__main__":
    main()