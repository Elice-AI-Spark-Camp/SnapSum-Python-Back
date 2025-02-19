from fastapi import FastAPI
from routes.crawling_routes import router as crawling_router
from routes.video_routes import router as video_router

app = FastAPI()

# 크롤링 API 라우트 등록
app.include_router(crawling_router, prefix="/crawl")
app.include_router(video_router, prefix="/video") 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)
