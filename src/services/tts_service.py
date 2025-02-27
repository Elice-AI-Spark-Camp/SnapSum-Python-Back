import os
import base64
import requests
import uuid
import json
from fastapi import HTTPException
from dotenv import load_dotenv, find_dotenv, dotenv_values

# 기존 환경 변수 초기화 (이미 로드된 값 무시)
os.environ.pop("TTS_API_KEY", None)
os.environ.pop("TTS_API_URL", None)
os.environ.pop("DOMAIN_URL", None)

# .env 파일 로드 - 절대 경로 사용
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# 환경변수에서 API 키와 도메인 가져오기
ELICE_TTS_API_KEY = os.getenv("TTS_API_KEY")
ELICE_TTS_API_URL = os.getenv("TTS_API_URL")
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

TTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tts_audio")
os.makedirs(TTS_DIR, exist_ok=True)

def generate_tts(text: str, language_code: str, voice_name: str) -> str:
    """
    Elice AI Spark Camp의 Google Cloud TTS API를 호출하여 음성 파일을 생성하고 저장한 후, 다운로드할 수 있는 URL 반환
    """
    payload = {
        "text": text,
        "voice": {
            "languageCode": language_code,
            "name": voice_name
        }
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": ELICE_TTS_API_KEY  # 환경 변수에서 직접 가져온 값 사용
    }

    # 디버깅을 위한 로그 추가
    print(f"TTS API 요청: {ELICE_TTS_API_URL}")
    print(f"TTS API 페이로드: {json.dumps(payload)}")
    response = requests.post(ELICE_TTS_API_URL, json=payload, headers=headers)

    # 디버깅을 위한 응답 로그 추가
    print(f"TTS API 응답 상태 코드: {response.status_code}")

    
    if response.status_code != 200:
        error_message = f"TTS 변환 실패: API 응답 오류 ({response.status_code})"
        try:
            error_detail = response.json()
            error_message += f" - {json.dumps(error_detail)}"
        except:
            error_message += f" - {response.text}"
        
        print(f"오류: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

    try:
        response_json = response.json()
        
        # 응답 구조 확인
        if "audioContent" in response_json:
            audio_content = response_json["audioContent"]
        else:
            # 응답 구조가 다를 수 있으므로 다른 키 확인
            possible_keys = ["audio", "audio_content", "content", "wav_bytes"]
            for key in possible_keys:
                if key in response_json:
                    audio_content = response_json[key]
                    print(f"오디오 데이터 키 발견: {key}")
                    break
            else:
                # 응답 구조 출력
                print(f"응답에서 오디오 데이터를 찾을 수 없음. 응답 키: {list(response_json.keys())}")
                raise HTTPException(status_code=500, detail=f"TTS 변환 실패: 응답에서 오디오 데이터 없음. 응답 키: {list(response_json.keys())}")
    except Exception as e:
        print(f"응답 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: 응답 처리 중 오류 - {str(e)}")

    if not audio_content:
        print("오디오 콘텐츠가 비어 있음")
        raise HTTPException(status_code=500, detail="TTS 변환 실패: 응답에서 오디오 데이터가 비어 있음")

    # UUID를 사용하여 고유한 파일명 생성
    unique_filename = f"tts_{uuid.uuid4()}.wav"
    audio_filename = f"{TTS_DIR}/{unique_filename}"
    
    try:
        # Base64 디코딩하여 음성 파일 저장
        audio_data = base64.b64decode(audio_content)
        with open(audio_filename, "wb") as audio_file:
            audio_file.write(audio_data)
        
        print(f"오디오 파일 저장 완료: {audio_filename}")
    except Exception as e:
        print(f"오디오 파일 저장 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: 오디오 파일 저장 중 오류 - {str(e)}")

    # 클라우드 환경의 도메인을 사용하여 URL 생성
    return f"{DOMAIN_URL}/tts_audio/{unique_filename}"
