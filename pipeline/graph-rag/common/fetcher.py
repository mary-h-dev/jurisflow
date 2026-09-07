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
    """Fetch an HTML page and return it as a BeautifulSoup object."""
    try:
        time.sleep(delay)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = "utf-8"
        return BeautifulSoup(response.text, "lxml")

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {url}.")
    except requests.exceptions.Timeout:
        print(f"❌ Timeout for {url}")

    return None


def save_raw_html(url: str, output_path: str) -> bool:
    """Save the raw HTML for offline use."""
    try:
        time.sleep(1.5)
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        if not response.text or len(response.text.strip()) == 0:
            print(f"⚠️ Server response for {url} was empty (status={response.status_code}) — not saved.")
            return False

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ HTML saved: {output_path} ({len(response.text)} characters)")
        return True
    except Exception as e:
        print(f"❌ Error saving HTML: {e}")
        return False


def load_local_html(file_path: str) -> BeautifulSoup | None:
    """Load HTML from a local file (offline mode)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return BeautifulSoup(f.read(), "lxml")
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None