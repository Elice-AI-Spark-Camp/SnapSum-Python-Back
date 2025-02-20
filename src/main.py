from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.crawling_routes import router as crawling_router
from routes.video_routes import router as video_router
import imageio_ffmpeg
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

# 비디오 저장 폴더 생성 (없으면 생성)
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


# 크롤링 API 라우트 등록
app.include_router(crawling_router, prefix="/crawl")
app.include_router(video_router, prefix="/video") 

# 정적 파일 서빙 설정 (비디오 파일)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)
