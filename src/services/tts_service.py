import os
import base64
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
ELICE_TTS_API_KEY = os.getenv("TTS_API_KEY")
ELICE_TTS_API_URL = os.getenv("TTS_API_URL")

TTS_DIR = "tts_audio"
os.makedirs(TTS_DIR, exist_ok=True)  # TTS 파일 저장 폴더 생성


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
    print(ELICE_TTS_API_KEY)
    print(ELICE_TTS_API_URL)


    response = requests.post(ELICE_TTS_API_URL, json=payload, headers=headers)
    print(response)

    # 응답 검증
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: API 응답 오류 ({response.status_code})")

    # 응답 JSON에서 'wav_bytes' 필드 추출
    response_json = response.json()
    wav_base64 = response_json.get("wav_bytes")

    if not wav_base64:
        raise HTTPException(status_code=500, detail="TTS 변환 실패: 응답에서 오디오 데이터 없음")

    # Base64 디코딩하여 음성 파일 저장
    audio_data = base64.b64decode(wav_base64)
    audio_filename = f"{TTS_DIR}/tts_output.wav"
    
    with open(audio_filename, "wb") as audio_file:
        audio_file.write(audio_data)

    return f"/tts/audio/tts_output.wav"  # 다운로드 가능한 파일 경로 반환
