"""
test_ara.py — تست ساده: آیا ara.jri.ac.ir با requests معمولی قابل دریافته یا نه
(بدون هیچ ذخیره‌ی دائمی — فقط یک تست سریع)
"""
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

url = "https://ara.jri.ac.ir/Judge/Text/32208"
r = requests.get(url, headers=HEADERS, timeout=15)

print(f"📏 حجم پاسخ: {len(r.text)} کاراکتر")
print(f"📌 status code: {r.status_code}")

# آیا متن واقعی رأی توش هست؟
if "طلاق" in r.text or "دادنامه" in r.text:
    print("✅ به‌نظر می‌رسه محتوای واقعی برگشته (requests معمولی کافیه)")
else:
    print("⚠️ محتوای واقعی پیدا نشد — احتمالاً نیاز به روش دیگه‌ای داریم")
    print("نمونه‌ی ۳۰۰ کاراکتر اول:")
    print(r.text[:300])