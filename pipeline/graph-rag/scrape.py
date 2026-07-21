import requests
import re
import json
from pathlib import Path

# ───────────────────────────────────────────────
# 1. دریافت صفحه
# ───────────────────────────────────────────────
url = "https://mahzarchi.ir/education/7128/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("⏳ در حال دریافت صفحه...")
response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
print(f"✅ صفحه دریافت شد | اندازه: {len(response.text):,} کاراکتر")

# ───────────────────────────────────────────────
# 2. استخراج آرایه JSON از اسکریپت
# ───────────────────────────────────────────────
print("\n⏳ در حال استخراج داده‌ها...")
match = re.search(r'const data = (\[.*?\]);', response.text, re.DOTALL)

if not match:
    print("❌ آرایه JSON پیدا نشد!")
    exit(1)

json_str = match.group(1)
data = json.loads(json_str)

print(f"✅ تعداد واژه‌های استخراج شده: {len(data)}")

# نمایش ۵ نمونه
print("\n📖 ۵ نمونه اول:")
for item in data[:5]:
    meaning_preview = item['معنی'][:70] + "..." if len(item['معنی']) > 70 else item['معنی']
    print(f"   • {item['واژه']}: {meaning_preview}")

# ───────────────────────────────────────────────
# 3. ذخیره به فرمت JSON
# ───────────────────────────────────────────────
output_file = Path("legal_vocabulary.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n💾 فایل ذخیره شد: {output_file.absolute()}")
print(f"   اندازه فایل: {output_file.stat().st_size:,} بایت")

# ───────────────────────────────────────────────
# 4. آماده‌سازی Closed Vocabulary برای LLM
# ───────────────────────────────────────────────
# فرمت جایگزین: لیست ساده برای استفاده در Prompt
vocab_list = [item['واژه'] for item in data]
vocab_text = "\n".join([f"- {item['واژه']}: {item['معنی']}" for item in data])

# ذخیره فرمت متنی (برای استفاده در Prompt)
text_file = Path("legal_vocabulary_prompt.txt")
with open(text_file, 'w', encoding='utf-8') as f:
    f.write("=== واژگان حقوقی (Closed Vocabulary) ===\n\n")
    f.write(vocab_text)

print(f"💾 فایل Prompt ذخیره شد: {text_file.absolute()}")

# آمار
print(f"\n📊 آمار:")
print(f"   • تعداد کل واژه‌ها: {len(data)}")
print(f"   • میانگین طول معنی: {sum(len(item['معنی']) for item in data) // len(data)} کاراکتر")