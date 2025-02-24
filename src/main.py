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
    title="SnapSum API",
    servers=[
        {"url": DOMAIN_URL, "description": "Current Environment"}
    ]
)

# 비디오 저장 폴더 생성 (없으면 생성)
VIDEO_DIR = "videos"
TTS_DIR = "tts_audio"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)


# 크롤링 API 라우트 등록
app.include_router(crawling_router, prefix="/crawl")
app.include_router(video_router, prefix="/video") 
app.include_router(tts_router, prefix="/tts")

# 정적 파일 서빙 설정 (비디오 파일)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/tts_audio", StaticFiles(directory=TTS_DIR), name="tts_audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)
