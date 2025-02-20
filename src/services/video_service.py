import os
from datetime import datetime
from moviepy.editor import TextClip, CompositeVideoClip

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

def generate_video_file(video_id: int) -> str:
    """MoviePy를 사용하여 비디오를 생성하는 함수"""

    # 비디오 파일 경로 설정
    video_filename = f"{VIDEO_DIR}/video_{video_id}.mp4"

    # 텍스트 기반의 간단한 영상 생성
    text_clip = TextClip(
        f"Video ID: {video_id}\nGenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=50, color="white", size=(1280, 720)
    ).set_duration(5)

    video = CompositeVideoClip([text_clip])
    video.write_videofile(video_filename, fps=24, codec="libx264")

    return video_filename