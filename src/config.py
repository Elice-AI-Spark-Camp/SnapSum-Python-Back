import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    TTS_API_URL = os.getenv("TTS_API_URL")
    TTS_API_KEY = os.getenv("TTS_API_KEY")
