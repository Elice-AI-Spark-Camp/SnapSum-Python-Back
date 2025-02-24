import os
import base64
import requests
import uuid
from fastapi import HTTPException
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경변수에서 API 키와 도메인 가져오기
ELICE_TTS_API_KEY = os.getenv("TTS_API_KEY")
ELICE_TTS_API_URL = os.getenv("TTS_API_URL")
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")  # 기본값 설정

TTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tts_audio")
os.makedirs(TTS_DIR, exist_ok=True)

def generate_tts(text: str, language_code: str, voice_name: str) -> str:
    """
    엘리스 TTS API를 호출하여 음성 파일을 생성하고 저장한 후, 다운로드할 수 있는 URL 반환
    """
    payload = {
        "text": text,
        "language_code": language_code,
        "voice_name": voice_name
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"{ELICE_TTS_API_KEY}"
    }

    response = requests.post(ELICE_TTS_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: API 응답 오류 ({response.status_code})")

    response_json = response.json()
    wav_base64 = response_json.get("wav_bytes")

    if not wav_base64:
        raise HTTPException(status_code=500, detail="TTS 변환 실패: 응답에서 오디오 데이터 없음")

    # UUID를 사용하여 고유한 파일명 생성
    unique_filename = f"tts_{uuid.uuid4()}.wav"
    audio_filename = f"{TTS_DIR}/{unique_filename}"
    
    # Base64 디코딩하여 음성 파일 저장
    audio_data = base64.b64decode(wav_base64)
    with open(audio_filename, "wb") as audio_file:
        audio_file.write(audio_data)

    # 클라우드 환경의 도메인을 사용하여 URL 생성
    return f"{DOMAIN_URL}/tts_audio/{unique_filename}"
