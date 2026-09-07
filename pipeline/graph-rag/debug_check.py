"""
debug_check.py — quick check whether the saved HTML actually has content
Run: python debug_check.py data/laws/civil_law.html
"""
import sys
from bs4 import BeautifulSoup

path = sys.argv[1] if len(sys.argv) > 1 else "data/laws/civil_law.html"

with open(path, "r", encoding="utf-8") as f:
    html = f.read()

print(f"📏 File size: {len(html)} characters")

soup = BeautifulSoup(html, "lxml")

# Check whether p.SecTex can be found at all
elements = soup.find_all("p", class_="SecTex")
print(f"🔍 p.SecTex elements found: {len(elements)}")

if len(elements) == 0:
    # Maybe a different class was used — show all p classes
    all_p = soup.find_all("p")
    print(f"📄 Total <p> tags: {len(all_p)}")
    classes_found = set()
    for p in all_p[:50]:
        if p.get("class"):
            classes_found.add(" ".join(p.get("class")))
    print(f"🏷️  p classes found (sample): {classes_found}")

    # Check whether the page returned a captcha or an error
    title = soup.find("title")
    print(f"📌 Page title: {title.get_text() if title else 'not found'}")
    print(f"📝 First 500 characters of page text:")
    print(soup.get_text()[:500])