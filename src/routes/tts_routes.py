from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from services.tts_service import generate_tts
from models.tts_models import TTSRequest, TTSResponse
import os

router = APIRouter()

@router.post("/generate", 
             summary="TTS 변환", 
             response_model=TTSResponse, 
             description="텍스트를 음성으로 변환하여 다운로드 가능한 URL 반환\n\n음성 옵션:\n- 남성 음성 1: ko-KR-Standard-C\n- 남성 음성 2: ko-KR-Standard-D\n- 여성 음성 1: ko-KR-Standard-A\n- 여성 음성 2: ko-KR-Standard-B",
             responses={
                 200: {
                     "description": "TTS 변환 성공",
                     "content": {
                         "application/json": {
                             "example": {
                                 "audio_url": "http://localhost:5001/tts_audio/tts_123e4567-e89b-12d3-a456-426614174000.wav"
                             }
                         }
                     }
                 },
                 500: {
                     "description": "서버 오류",
                     "content": {
                         "application/json": {
                             "example": {
                                 "detail": "TTS 변환 실패: API 응답 오류 (500)"
                             }
                         }
                     }
                 }
             })
async def tts_generate(request: TTSRequest):
    """
    텍스트를 음성으로 변환하여 다운로드 가능한 URL을 반환합니다.
    
    Google Cloud TTS API를 사용하여 텍스트를 음성으로 변환하고,
    생성된 음성 파일은 서버에 저장됩니다.
    
    다양한 음성 옵션을 선택할 수 있으며, 기본값은 한국어 남성 음성입니다.
    """
    try:
        audio_url = await generate_tts(
            request.text, request.language_code, request.voice_name)
        return TTSResponse(audio_url=audio_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {str(e)}")
