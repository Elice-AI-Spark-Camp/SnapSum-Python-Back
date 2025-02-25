from fastapi import APIRouter, HTTPException
from services.video_service import generate_video_with_tts_and_images
from models.video_models import VideoRequest, VideoResponse

router = APIRouter()

@router.post("/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """비디오 생성 API"""
    if request.summaryId <= 0:
        raise HTTPException(status_code=400, detail="유효하지 않은 summaryId")
    
    # 비디오 생성 서비스 호출
    video_url = await generate_video_with_tts_and_images(
        request.summaryId,
        request.paragraphs,
        request.voiceId,
        request.imageUrls
    )
    
    return VideoResponse(videoId=request.summaryId, status="COMPLETED", videoUrl=video_url)