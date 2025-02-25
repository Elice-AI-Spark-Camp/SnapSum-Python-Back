from pydantic import BaseModel, Field
from typing import List, Dict

class Paragraph(BaseModel):
    text: str = Field(..., description="문단 텍스트", example="첫 번째 문단입니다. 이 문단은 비디오의 첫 부분에 나타납니다.")
    index: int = Field(..., description="문단 인덱스", example=0)

class VideoRequest(BaseModel):
    summaryId: int = Field(..., description="요약 ID", example=1)
    paragraphs: List[str] = Field(..., description="문단 텍스트 목록", 
                                 example=["첫 번째 문단입니다.", "두 번째 문단입니다."])
    voice: str = Field(..., description="TTS 음성 ID", example="ko-KR-Standard-A")
    imageUrls: Dict[str, str] = Field(..., description="문단 인덱스와 이미지 URL 매핑", 
                                     example={"0": "https://picsum.photos/800/600", 
                                              "1": "https://picsum.photos/800/600?random=1"})
    
    class Config:
        schema_extra = {
            "example": {
                "summaryId": 1,
                "paragraphs": ["첫 번째 문단입니다.", "두 번째 문단입니다."],
                "voice": "ko-KR-Standard-A",
                "imageUrls": {
                    "0": "https://picsum.photos/800/600",
                    "1": "https://picsum.photos/800/600?random=1"
                }
            }
        }

class VideoResponse(BaseModel):
    videoId: int = Field(..., description="비디오 ID", example=1)
    status: str = Field(..., description="비디오 생성 상태", example="COMPLETED")
    videoUrl: str = Field(..., description="비디오 URL", 
                         example="http://localhost:5001/videos/video_123e4567-e89b-12d3-a456-426614174000.mp4")
    
    class Config:
        schema_extra = {
            "example": {
                "videoId": 1,
                "status": "COMPLETED",
                "videoUrl": "http://localhost:5001/videos/video_123e4567-e89b-12d3-a456-426614174000.mp4"
            }
        }