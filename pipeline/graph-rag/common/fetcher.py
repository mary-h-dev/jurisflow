

import time
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_page(url: str, delay: float = 1.5) -> BeautifulSoup | None:
    """دریافت صفحه HTML و برگرداندن BeautifulSoup"""
    try:
        time.sleep(delay)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = "utf-8"
        return BeautifulSoup(response.text, "lxml")

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
    except requests.exceptions.ConnectionError:
        print(f"❌ اتصال به {url} برقرار نشد.")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout برای {url}")

    return None


def save_raw_html(url: str, output_path: str) -> bool:
    """ذخیره HTML خام برای استفاده آفلاین"""
    try:
        time.sleep(1.5)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        if not response.text or len(response.text.strip()) == 0:
            print(f"⚠️ پاسخ سرور برای {url} خالی بود (status={response.status_code}) — ذخیره نشد.")
            return False

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ HTML ذخیره شد: {output_path} ({len(response.text)} کاراکتر)")
        return True
    except Exception as e:
        print(f"❌ خطا در ذخیره HTML: {e}")
        return False


def load_local_html(file_path: str) -> BeautifulSoup | None:
    """بارگذاری HTML از فایل محلی (حالت آفلاین)"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return BeautifulSoup(f.read(), "lxml")
    except FileNotFoundError:
        print(f"❌ فایل یافت نشد: {file_path}")
        return None