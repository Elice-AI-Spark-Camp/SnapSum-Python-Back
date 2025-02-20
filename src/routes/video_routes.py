from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.video_service import generate_video_file
import os

router = APIRouter()

class VideoRequest(BaseModel):
    summaryId: int

class VideoResponse(BaseModel):
    videoId: int
    status: str
    videoUrl: str

@router.post("/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """비디오 생성 API"""
    if request.summaryId <= 0:
        raise HTTPException(status_code=400, detail="유효하지 않은 summaryId")
    
    # VideoService 호출하여 비디오 생성
    video_id = request.summaryId  # 실제로는 DB에서 생성된 ID 사용
    video_path = generate_video_file(video_id)

    video_url = f"http://localhost:5001/videos/{os.path.basename(video_path)}"
    
    return VideoResponse(videoId=video_id, status="COMPLETED", videoUrl=video_url)