import os
import uuid
import aiohttp
import tempfile
from typing import List, Dict
from datetime import datetime
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from dotenv import load_dotenv
from services.tts_service import generate_tts
from models.video_models import Paragraph

# .env 로드
load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

async def download_image(url: str, temp_dir: str) -> str:
    """이미지 URL에서 이미지 다운로드"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"이미지 다운로드 실패: {url}, 상태 코드: {response.status}")
            
            # 임시 파일 생성
            filename = os.path.join(temp_dir, f"image_{uuid.uuid4()}.jpg")
            with open(filename, 'wb') as f:
                f.write(await response.read())
            
            return filename

async def generate_video_with_tts_and_images(
    summary_id: int,
    paragraphs: List[Paragraph],
    voice_id: str,
    image_urls: Dict[str, str]
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성"""
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 비디오 클립 목록
        video_clips = []
        
        # 각 문단별로 처리
        for paragraph in paragraphs:
            # 1. TTS 생성
            audio_url = generate_tts(
                text=paragraph.text,
                language_code="ko-KR",  # 한국어 기본값
                voice_name=voice_id
            )
            
            # TTS 파일 경로 추출 (URL에서 파일명 추출)
            audio_filename = audio_url.split('/')[-1]
            audio_path = os.path.join("tts_audio", audio_filename)
            
            # 2. 이미지 다운로드
            image_url = image_urls.get(str(paragraph.index))
            if not image_url:
                raise Exception(f"문단 {paragraph.index}의 이미지 URL이 없습니다.")
            
            image_path = await download_image(image_url, temp_dir)
            
            # 3. 오디오 길이 가져오기
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # 4. 이미지 클립 생성 및 오디오 추가
            image_clip = ImageClip(image_path, duration=audio_duration)
            video_clip = image_clip.set_audio(audio_clip)
            
            video_clips.append(video_clip)
        
        # 모든 비디오 클립 연결
        final_clip = concatenate_videoclips(video_clips)
        
        # UUID를 사용하여 고유한 파일명 생성
        unique_filename = f"video_{uuid.uuid4()}.mp4"
        video_filename = f"{VIDEO_DIR}/{unique_filename}"
        
        # 비디오 파일 생성
        final_clip.write_videofile(
            video_filename, 
            fps=24, 
            codec="libx264",
            audio_codec="aac"
        )
        
        # 임시 파일 정리는 tempfile.TemporaryDirectory가 자동으로 처리
        
        return f"{DOMAIN_URL}/videos/{unique_filename}"

def generate_video_file(video_id: int) -> str:
    """MoviePy를 사용하여 비디오를 생성하는 함수"""

    # UUID를 사용하여 고유한 파일명 생성
    unique_filename = f"video_{uuid.uuid4()}.mp4"
    video_filename = f"{VIDEO_DIR}/{unique_filename}"

    # 텍스트 기반의 간단한 영상 생성
    text_clip = TextClip(
        f"Video ID: {video_id}\nGenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=50, color="white", size=(1280, 720)
    ).set_duration(5)

    video = CompositeVideoClip([text_clip])
    video.write_videofile(video_filename, fps=24, codec="libx264")

    return f"{DOMAIN_URL}/videos/{unique_filename}"