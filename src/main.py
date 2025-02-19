from flask import Flask
from routes.crawling_routes import crawling_bp

app = Flask(__name__)

app.register_blueprint(crawling_bp, url_prefix="/crawl")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
