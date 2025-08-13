# loadbot/config.py
import os

# OPTIONAL: allow .env locally
try:
    from dotenv import load_dotenv  # add "python-dotenv" to requirements.txt if you use this
    load_dotenv()
except Exception:
    pass

# Read token from env (Railway UI Variables)
BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")  # Railway variable name
    or os.getenv("BOT_TOKEN")        # fallback if you use this name locally
)

OPENAI_API_KEY = (
    os.getenv("OPEN_AI_KEY")         # your Railway var in the screenshot
    or os.getenv("OPENAI_API_KEY")   # common alternative
)

# Optional safety: fail fast if missing
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing. Set TELEGRAM_BOT_TOKEN (or BOT_TOKEN).")