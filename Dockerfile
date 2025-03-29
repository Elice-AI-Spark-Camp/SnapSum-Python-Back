FROM python:3.10-slim

# FFmpeg 설치
RUN apt update && apt install -y ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
