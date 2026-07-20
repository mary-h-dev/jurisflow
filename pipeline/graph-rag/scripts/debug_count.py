import requests

url = "https://ara.jri.ac.ir/Judge/Index?Laws=2"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

response = requests.get(url, headers=headers, timeout=15)
response.encoding = "utf-8"

# ذخیره ۵۰۰۰ کاراکتر اول و آخر صفحه برای بررسی متن لود شده
with open("test_page.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("✅ ۵۰۰۰ کاراکتر اول صفحه ذخیره شد. حالا کد زیر را اجرا کن تا ببینیم عبارت 'یافته' چطور در صفحه ذخیره شده:")

# پیدا کردن موقعیت کلمه‌ی "یافته" در متن دانلود شده
text = response.text
index = text.find("یافته")
if index != -1:
    print("\n📌 متن اطراف کلمه‌ی پیدا شده:")
    print(text[max(0, index-100): index+100])
else:
    print("\n❌ متأسفانه کلمه‌ی 'یافته' اصلاً در متن صفحه پیدا نشد!")