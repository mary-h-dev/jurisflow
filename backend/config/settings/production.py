from .base import *
import os
from datetime import timedelta

# Core
SECRET_KEY    = os.environ["DJANGO_SECRET_KEY"]
DEBUG         = False
ALLOWED_HOSTS = os.environ["ALLOWED_HOSTS"].split(",")

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME":     os.environ["DB_NAME"],
        "USER":     os.environ["DB_USER"],
        "PASSWORD": os.environ["DB_PASSWORD"],
        "HOST":     os.environ["DB_HOST"],
        "PORT":     os.environ.get("DB_PORT", "5432"),
    }
}

# CORS
CORS_ALLOWED_ORIGINS = os.environ["CORS_ALLOWED_ORIGINS"].split(",")

# Security
SECURE_SSL_REDIRECT            = True
SESSION_COOKIE_SECURE          = True
CSRF_COOKIE_SECURE             = True
SECURE_HSTS_SECONDS            = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True

# JWT
JWT_SECRET_KEY       = SECRET_KEY
JWT_ACCESS_LIFETIME  = timedelta(minutes=30)
JWT_REFRESH_LIFETIME = timedelta(days=7)
JWT_ALGORITHM        = "HS256"

# Neo4j
NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# Groq
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Gemini
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Ollama
OLLAMA_URL   = os.environ["OLLAMA_URL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]




AUDITOR_LLM_PROVIDER = "openrouter"
AUDITOR_LLM_REQUEST_DELAY_SECONDS = 4.0  

OPENROUTER_API_KEY=os.environ["OPENROUTER_API_KEY"]