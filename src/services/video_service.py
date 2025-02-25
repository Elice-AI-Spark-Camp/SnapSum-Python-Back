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
from fastapi import HTTPException

# .env 로드
load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

# 비디오 저장 디렉토리
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# 대체 이미지 경로
DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "default_image.jpg")

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
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str]
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성"""
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        clips = []
        
        # 각 문단별로 처리
        for i, paragraph in enumerate(paragraphs):
            # 문단 인덱스를 문자열로 변환
            idx_str = str(i)
            
            # 해당 문단의 이미지 URL 가져오기
            image_url = image_urls.get(idx_str)
            
            if not image_url:
                print(f"문단 {idx_str}에 대한 이미지 URL이 없습니다.")
                continue
            
            # TTS 생성
            try:
                audio_url = generate_tts(paragraph, "ko-KR", voice_id)
                audio_path = audio_url.replace(f"{DOMAIN_URL}/", "")
                
                # 이미지 다운로드
                image_path = os.path.join(temp_dir, f"image_{i}.jpg")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                with open(image_path, "wb") as f:
                                    f.write(await response.read())
                            else:
                                print(f"이미지 다운로드 실패: {image_url}, 상태 코드: {response.status}")
                                # 대체 이미지 사용
                                if os.path.exists(DEFAULT_IMAGE_PATH):
                                    image_path = DEFAULT_IMAGE_PATH
                                else:
                                    # 텍스트 클립 생성
                                    text_clip = TextClip(
                                        f"이미지를 찾을 수 없습니다.\n\n{paragraph}",
                                        fontsize=30, color="white", bg_color="black",
                                        size=(1280, 720), method="caption"
                                    ).set_duration(5)
                                    
                                    # 오디오 클립 생성
                                    audio_clip = AudioFileClip(audio_path)
                                    
                                    # 텍스트 클립에 오디오 추가
                                    video_clip = text_clip.set_audio(audio_clip)
                                    video_clip = video_clip.set_duration(audio_clip.duration)
                                    
                                    clips.append(video_clip)
                                    continue
                except Exception as e:
                    print(f"이미지 다운로드 중 오류 발생: {str(e)}")
                    # 대체 이미지 또는 텍스트 클립 사용
                    if os.path.exists(DEFAULT_IMAGE_PATH):
                        image_path = DEFAULT_IMAGE_PATH
                    else:
                        # 텍스트 클립 생성
                        text_clip = TextClip(
                            f"이미지를 찾을 수 없습니다.\n\n{paragraph}",
                            fontsize=30, color="white", bg_color="black",
                            size=(1280, 720), method="caption"
                        ).set_duration(5)
                        
                        # 오디오 클립 생성
                        audio_clip = AudioFileClip(audio_path)
                        
                        # 텍스트 클립에 오디오 추가
                        video_clip = text_clip.set_audio(audio_clip)
                        video_clip = video_clip.set_duration(audio_clip.duration)
                        
                        clips.append(video_clip)
                        continue
                
                # 이미지 클립 생성
                image_clip = ImageClip(image_path).set_duration(5)
                
                # 오디오 클립 생성
                audio_clip = AudioFileClip(audio_path)
                
                # 이미지 클립에 오디오 추가
                video_clip = image_clip.set_audio(audio_clip)
                
                # 오디오 길이에 맞게 비디오 길이 조정
                video_clip = video_clip.set_duration(audio_clip.duration)
                
                clips.append(video_clip)
                
            except Exception as e:
                print(f"TTS 생성 중 오류 발생: {str(e)}")
                # 오류 발생 시 텍스트 클립 생성
                text_clip = TextClip(
                    f"오디오를 생성할 수 없습니다.\n\n{paragraph}",
                    fontsize=30, color="white", bg_color="black",
                    size=(1280, 720), method="caption"
                ).set_duration(5)
                
                clips.append(text_clip)
        
        if not clips:
            # 클립이 없는 경우 기본 텍스트 클립 생성
            text_clip = TextClip(
                "비디오를 생성할 수 없습니다.\n\n문단이 없거나 모든 문단 처리에 실패했습니다.",
                fontsize=30, color="white", bg_color="black",
                size=(1280, 720), method="caption"
            ).set_duration(5)
            
            clips.append(text_clip)
        
        # 모든 클립 연결
        final_clip = concatenate_videoclips(clips)
        
        # 고유한 파일명 생성
        unique_filename = f"video_{uuid.uuid4()}.mp4"
        video_filename = os.path.join(VIDEO_DIR, unique_filename)
        
        # 비디오 파일 생성
        final_clip.write_videofile(
            video_filename, 
            fps=24, 
            codec="libx264",
            audio_codec="aac"
        )
        
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