from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.crawling_service import get_blog_content

router = APIRouter()


class BlogRequest(BaseModel):
    url: str


@router.post("/")
async def crawl(data: BlogRequest):
    blog_url = data.url

    if not blog_url or "blog.naver.com" not in blog_url:
        raise HTTPException(status_code=400, detail="❌ 올바른 네이버 블로그 링크가 아님")

    blog_content = get_blog_content(blog_url)
    return {"blog_content": blog_content}
