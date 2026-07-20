"""
find_form_and_test.py — پیدا کردن فرم واقعی صفحه‌بندی و امتحان کردنش با یک درخواست

هدف: کمینه کردن رفت‌وبرگشت (برای اینترنت کند) — همه‌چیز در یک اجرا.

اجرا:
    python find_form_and_test.py 867
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


def extract_ruling_ids(html):
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=lambda h: h and "/Judge/Text/" in h)
    ids = []
    for a in links:
        i = a.get("href").rstrip("/").split("/")[-1]
        if i not in ids:
            ids.append(i)
    return ids


def main():
    law_id = sys.argv[1]
    base_url = f"https://ara.jri.ac.ir/Judge/Index?Laws={law_id}"

    session = requests.Session()
    r = session.get(base_url, headers=HEADERS, timeout=20)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    page1_ids = extract_ruling_ids(r.text)
    print(f"صفحه‌ی اول: {len(page1_ids)} لینک")

    # پیدا کردن input فیلد PageNumber و بالا رفتن تا فرم دربرگیرنده‌اش
    page_input = soup.find("input", id="PageNumber")
    if not page_input:
        print("❌ input#PageNumber پیدا نشد")
        return

    form = page_input.find_parent("form")
    if not form:
        print("❌ فرم دربرگیرنده پیدا نشد")
        return

    action = form.get("action") or base_url
    method = (form.get("method") or "get").lower()
    print(f"📋 فرم: action={action}   method={method}")

    # جمع‌آوری همه‌ی فیلدهای این فرم با مقادیر پیش‌فرضشون
    form_data = {}
    for inp in form.find_all(["input", "select"]):
        name = inp.get("name")
        if not name:
            continue
        if inp.name == "select":
            selected = inp.find("option", selected=True) or inp.find("option")
            form_data[name] = selected.get("value") if selected else ""
        else:
            form_data[name] = inp.get("value", "")

    print(f"📦 فیلدهای فرم: {form_data}")

    # حالا PageNumber رو به ۲ تغییر بده و امتحان کن
    form_data["PageNumber"] = "2"

    full_action = action if action.startswith("http") else f"https://ara.jri.ac.ir{action}"

    if method == "post":
        r2 = session.post(full_action, data=form_data, headers=HEADERS, timeout=20)
    else:
        r2 = session.get(full_action, params=form_data, headers=HEADERS, timeout=20)

    r2.encoding = "utf-8"
    page2_ids = extract_ruling_ids(r2.text)
    print(f"\nبعد از ارسال کامل فرم با PageNumber=2: {len(page2_ids)} لینک")
    print(f"نمونه: {page2_ids[:3]}")
    print(f"آیا با صفحه‌ی اول فرق داره؟ {'بله ✅' if set(page2_ids) != set(page1_ids) else 'نه ❌'}")


if __name__ == "__main__":
    main()