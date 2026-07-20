"""
test_parse_case.py — تست کامل پارسر پرونده روی یک URL واقعی

اجرا:
    python test_parse_case.py https://ara.jri.ac.ir/Judge/Text/38508
"""

import sys
import json
import dataclasses
import requests
from bs4 import BeautifulSoup
from cases.parser import parse_ruling


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
    ruling_id = url.rstrip("/").split("/")[-1]

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    ruling = parse_ruling(soup, ruling_id, url)

    data = dataclasses.asdict(ruling)

    print(f"عنوان: {data['title']}")
    print(f"پیام: {data['summary'][:150]}...")
    print(f"گروه: {data['case_type']}   شماره دادنامه: {data['verdict_number']}   تاریخ: {data['verdict_date']}")
    print(f"مواد مستندشده: {data['cited_articles']}")
    print(f"پرونده‌های مرتبط: {data['related_ruling_ids']}")
    print(f"\nتعداد بخش‌های تفکیک‌شده: {len(data['sections'])}")
    for i, s in enumerate(data["sections"]):
        print(f"  {i+1}) {s['court_level']}  (طول متن: {len(s['text'])} کاراکتر)")

    print("\n--- خروجی کامل JSON ---")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()