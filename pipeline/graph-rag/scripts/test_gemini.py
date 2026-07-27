import os
from dotenv import load_dotenv
from google import genai

# خواندن فایل .env
load_dotenv()

# حالا کلاینت خودکار کلید را از فایل .env می‌خواند
client = genai.Client()

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="سلام! اتصال برقرار شد؟",
    )
    print(response.text)
except Exception as e:
    print(f"خطا: {e}")