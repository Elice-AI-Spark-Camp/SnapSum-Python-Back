# from flask import Flask
# from routes.crawling_routes import crawling_bp

# app = Flask(__name__)

# app.register_blueprint(crawling_bp, url_prefix="/crawl")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)

from fastapi import FastAPI
from routes.crawling_routes import router as crawling_router

app = FastAPI()

# 크롤링 API 라우트 등록
app.include_router(crawling_router, prefix="/crawl")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)

