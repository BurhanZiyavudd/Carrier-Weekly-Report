import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TOKEN")
AI_TOKEN = os.getenv("OPEN_AI_KEY")