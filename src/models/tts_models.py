from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    text: str = "안녕하세요. 반갑습니다."
    language_code: str = "ko-KR"
    voice_name: str = "ko-KR-Standard-C"

class TTSResponse(BaseModel):
    audio_url: str 