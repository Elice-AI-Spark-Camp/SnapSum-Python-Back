from flask import Blueprint, request, jsonify
from services.crawling_service import get_blog_content

crawling_bp = Blueprint("crawling", __name__)


@crawling_bp.route("/", methods=["POST"])
def crawl():
    data = request.get_json()
    blog_url = data.get("url")

    if not blog_url or "blog.naver.com" not in blog_url:
        return jsonify({"error": "❌ 올바른 네이버 블로그 링크가 아님"}), 400

    blog_content = get_blog_content(blog_url)
    return jsonify({"blog_content": blog_content})
