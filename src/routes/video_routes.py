from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class VideoRequest(BaseModel):
    summaryId: int

class VideoResponse(BaseModel):
    videoId: int
    status: str

@router.post("/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    if request.summaryId <= 0:
        raise HTTPException(status_code=400, detail="유효하지 않은 summary_id")
    
    videoId = request.summaryId  # 실제로는 DB에서 ID를 생성해야 함
    return VideoResponse(videoId=videoId, status="PROCESSING")
