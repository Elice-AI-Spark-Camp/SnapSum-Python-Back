from pydantic import BaseModel
from typing import List, Dict

class Paragraph(BaseModel):
    text: str
    index: int

class VideoRequest(BaseModel):
    summaryId: int
    paragraphs: List[Paragraph]
    voiceId: str
    imageUrls: Dict[str, str]  # 문단 인덱스(문자열)와 이미지 URL 매핑

class VideoResponse(BaseModel):
    videoId: int
    status: str
    videoUrl: str