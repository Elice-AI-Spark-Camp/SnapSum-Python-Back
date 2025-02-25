from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.crawling_routes import router as crawling_router
from routes.video_routes import router as video_router
from routes.tts_routes import router as tts_router 
import imageio_ffmpeg
import os
from dotenv import load_dotenv

load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(
    title="SnapSum Media API",
    description="텍스트 요약 및 비디오 생성을 위한 API 서비스",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    servers=[
        {"url": DOMAIN_URL, "description": "Current Environment"}
    ]
)

# 비디오 저장 폴더 생성 (없으면 생성)
VIDEO_DIR = "videos"
TTS_DIR = "tts_audio"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)


# API 라우트 등록
app.include_router(crawling_router, prefix="/crawl", tags=["Crawling"])
app.include_router(video_router, prefix="/video", tags=["Video"]) 
app.include_router(tts_router, prefix="/tts", tags=["TTS"])

# 정적 파일 서빙 설정 (비디오 파일)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/tts_audio", StaticFiles(directory=TTS_DIR), name="tts_audio")

@app.get("/", 
         tags=["Health Check"], 
         summary="API 서버 상태 확인", 
         description="API 서버가 정상적으로 실행 중인지 확인합니다.",
         responses={
             200: {
                 "description": "서버 정상 작동 중",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "online",
                             "message": "SnapSum Media API 서버가 실행 중입니다.",
                             "version": "1.0.0"
                         }
                     }
                 }
             }
         })
async def root():
    """
    API 서버의 상태를 확인합니다.
    
    이 엔드포인트는 서버가 정상적으로 실행 중인지 확인하는 데 사용됩니다.
    """
    return {
        "status": "online", 
        "message": "SnapSum Media API 서버가 실행 중입니다.",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)
