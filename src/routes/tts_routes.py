from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.tts_service import generate_tts
from fastapi.responses import FileResponse

router = APIRouter()


class TTSRequest(BaseModel):
    text: str = "안녕하세요. 반갑습니다."
    language_code: str = "ko-KR"
    voice_name: str = "ko-KR-Standard-C"


@router.post("/generate", summary="TTS 변환", description="텍스트를 음성으로 변환하여 다운로드 가능한 URL 반환\n\n음성 옵션:\n- 남성 음성 1: ko-KR-Standard-C\n- 남성 음성 2: ko-KR-Standard-D\n- 여성 음성 1: ko-KR-Standard-A\n- 여성 음성 2: ko-KR-Standard-B")
async def tts_generate(request: TTSRequest):
    try:
        audio_url = generate_tts(request.text, request.language_code, request.voice_name)
        return {"audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {str(e)}")


@router.get("/audio/{filename}", summary="TTS 파일 다운로드", description="생성된 TTS 파일을 다운로드")
async def tts_audio(filename: str):
    file_path = f"tts_audio/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없음")
    
    return FileResponse(file_path, media_type="audio/wav", filename=filename)
