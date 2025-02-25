from pydantic import BaseModel, Field
from enum import Enum

class VoiceName(str, Enum):
    male_1 = "ko-KR-Standard-C"  # 남성 음성 1
    male_2 = "ko-KR-Standard-D"  # 남성 음성 2
    female_1 = "ko-KR-Standard-A"  # 여성 음성 1
    female_2 = "ko-KR-Standard-B"  # 여성 음성 2

class TTSRequest(BaseModel):
    text: str = Field(
        default="안녕하세요. 반갑습니다.", 
        description="TTS로 변환할 텍스트",
        example="안녕하세요. 이 텍스트는 음성으로 변환됩니다."
    )
    language_code: str = Field(
        default="ko-KR", 
        description="언어 코드",
        example="ko-KR"
    )
    voice_name: str = Field(
        default="ko-KR-Standard-C", 
        description="음성 ID (ko-KR-Standard-A, ko-KR-Standard-B, ko-KR-Standard-C, ko-KR-Standard-D)",
        example="ko-KR-Standard-C"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "안녕하세요. 이 텍스트는 음성으로 변환됩니다.",
                "language_code": "ko-KR",
                "voice_name": "ko-KR-Standard-C"
            }
        }

class TTSResponse(BaseModel):
    audio_url: str = Field(
        ..., 
        description="생성된 오디오 파일 URL",
        example="http://localhost:5001/tts_audio/tts_123e4567-e89b-12d3-a456-426614174000.wav"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "audio_url": "http://localhost:5001/tts_audio/tts_123e4567-e89b-12d3-a456-426614174000.wav"
            }
        } 