from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from services.video_service import generate_video_with_tts_and_images
from models.video_models import VideoRequest, VideoResponse

router = APIRouter()


@router.post("/generate", 
             summary="비디오 생성", 
             response_model=VideoResponse, 
             description="문단별 텍스트와 이미지를 사용하여 TTS 음성이 포함된 비디오를 생성합니다.\n\n"
                        "- summaryId: 요약 ID (스프링 백엔드에서 제공)\n"
                        "- paragraphs: 문단별 텍스트 목록\n"
                        "- voice: TTS 음성 ID (ko-KR-Standard-A, ko-KR-Standard-B, ko-KR-Standard-C, ko-KR-Standard-D)\n"
                        "- imageUrls: 문단 인덱스와 이미지 URL 매핑",
             responses={
                 200: {
                     "description": "비디오 생성 성공",
                     "content": {
                         "application/json": {
                             "example": {
                                 "videoId": 1,
                                 "status": "COMPLETED",
                                 "videoUrl": "http://localhost:5001/videos/video_123e4567-e89b-12d3-a456-426614174000.mp4"
                             }
                         }
                     }
                 },
                 400: {
                     "description": "잘못된 요청",
                     "content": {
                         "application/json": {
                             "example": {
                                 "detail": "유효하지 않은 summaryId"
                             }
                         }
                     }
                 },
                 500: {
                     "description": "서버 오류",
                     "content": {
                         "application/json": {
                             "example": {
                                 "detail": "비디오 생성 중 오류가 발생했습니다: 이미지 다운로드 실패"
                             }
                         }
                     }
                 }
             })
async def generate_video(request: VideoRequest):
    """
    문단별 텍스트와 이미지를 사용하여 TTS 음성이 포함된 비디오를 생성합니다.
    
    각 문단별로 TTS를 생성하고, 해당 이미지와 결합하여 비디오 클립을 만든 후,
    모든 클립을 연결하여 최종 비디오를 생성합니다.
    
    생성된 비디오는 서버에 저장되며, 다운로드 가능한 URL이 반환됩니다.
    """
    if request.summaryId <= 0:
        raise HTTPException(status_code=400, detail="유효하지 않은 summaryId")
    
    # 비디오 생성 서비스 호출
    video_url = await generate_video_with_tts_and_images(
        request.summaryId,
        request.paragraphs,
        request.voice,
        request.imageUrls
    )
    
    return VideoResponse(videoId=request.summaryId, status="COMPLETED", videoUrl=video_url)