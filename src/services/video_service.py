import os
import uuid
import aiohttp
import tempfile
import re
import asyncio
import subprocess
from typing import List, Dict
from datetime import datetime
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from dotenv import load_dotenv
from services.tts_service import generate_tts
from models.video_models import Paragraph
from fastapi import HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# .env 로드
load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

# 비디오 저장 디렉토리
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# 대체 이미지 경로
DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "default_image.jpg")

# 나눔 고딕 폰트 경로 설정
NANUM_GOTHIC_FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
print(f"나눔 고딕 폰트 경로: {NANUM_GOTHIC_FONT}")
print(f"폰트 파일 존재 여부: {os.path.exists(NANUM_GOTHIC_FONT)}")

class Paragraph(BaseModel):
    text: str
    image_url: str

def split_paragraph_into_sentences(paragraph):
    """문단을 문장으로 분리하는 함수"""
    # 한국어 문장 분리 패턴 (마침표, 느낌표, 물음표 뒤에 공백이 있는 경우)
    sentences = re.split(r'([.!?]\s)', paragraph)
    result = []
    i = 0
    
    while i < len(sentences) - 1:
        if i + 1 < len(sentences) and re.match(r'[.!?]\s', sentences[i + 1]):
            result.append(sentences[i] + sentences[i + 1])
            i += 2
        else:
            result.append(sentences[i])
            i += 1
    
    if i < len(sentences):
        result.append(sentences[i])
    
    # 빈 문자열 제거 및 공백 제거
    result = [s.strip() for s in result if s.strip()]
    
    # 결과가 없으면 원본 문단을 그대로 반환
    if not result:
        return [paragraph]
    
    return result

def calculate_sentence_durations(sentences, total_duration):
    """각 문장의 예상 지속 시간을 계산하는 함수"""
    total_chars = sum(len(s) for s in sentences)
    durations = []
    
    for sentence in sentences:
        # 문장 길이에 비례하여 시간 할당 (최소 1초)
        ratio = len(sentence) / total_chars if total_chars > 0 else 1 / len(sentences)
        duration = max(ratio * total_duration, 1.0)
        durations.append(duration)
    
    return durations

async def download_image(session, url, dest_path):
    """이미지 URL에서 이미지 다운로드"""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: HTTP {response.status}")
            
            with open(dest_path, 'wb') as f:
                f.write(await response.read())
            
            return dest_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: {str(e)}")

def create_subtitle_image(text, width, height, font_size=30, font_path=NANUM_GOTHIC_FONT):
    """PIL을 사용하여 자막 이미지 생성"""
    try:
        # 투명한 배경의 이미지 생성
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 폰트 로드
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"폰트 로드 성공: {font_path}")
        except Exception as e:
            print(f"폰트 로드 오류: {str(e)}")
            # 기본 폰트 사용
            print("기본 폰트 사용")
            font = None
        
        # 텍스트 크기 계산
        try:
            if font:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                print(f"텍스트 크기: {text_width}x{text_height}")
            else:
                # 폰트가 없는 경우 대략적인 크기 추정
                text_width = len(text) * font_size // 2
                text_height = font_size
        except Exception as e:
            print(f"텍스트 크기 계산 오류: {str(e)}")
            # 기본값 사용
            text_width = min(len(text) * font_size // 2, width - 40)
            text_height = font_size
            print(f"텍스트 크기 기본값 사용: {text_width}x{text_height}")
        
        # 텍스트 배경 영역 계산
        padding = 20
        bg_width = min(text_width + padding * 2, width - 40)
        bg_height = text_height + padding
        
        # 텍스트 위치 (하단 중앙)
        bg_left = (width - bg_width) // 2
        bg_top = height - bg_height - 30
        bg_right = bg_left + bg_width
        bg_bottom = bg_top + bg_height
        
        # 배경 그리기 (반투명 검정)
        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=(0, 0, 0, 180))
        
        # 텍스트 그리기 (흰색)
        if font:
            text_x = bg_left + padding
            text_y = bg_top + (bg_height - text_height) // 2
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
        
        # PIL 이미지를 numpy 배열로 변환
        return np.array(img)
    
    except Exception as e:
        print(f"자막 이미지 생성 오류: {str(e)}")
        # 오류 발생 시 빈 이미지 반환
        return np.zeros((height, width, 4), dtype=np.uint8)

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
            
            try:
                # 이미지 다운로드
                image_path = os.path.join(temp_dir, f"image_{i}.jpg")
                async with aiohttp.ClientSession() as session:
                    await download_image(session, image_url, image_path)
                
                # TTS 생성
                audio_url = generate_tts(paragraph, "ko-KR", voice_id)
                
                # 오디오 다운로드
                audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
                async with aiohttp.ClientSession() as session:
                    await download_image(session, audio_url, audio_path)
                
                # 이미지 클립 생성
                image_clip = ImageClip(image_path)
                
                # 오디오 클립 생성
                audio_clip = AudioFileClip(audio_path)
                
                # 이미지 클립에 오디오 추가
                video_clip = image_clip.set_audio(audio_clip)
                
                # 오디오 길이에 맞게 비디오 길이 조정
                video_clip = video_clip.set_duration(audio_clip.duration)
                
                # PIL을 사용하여 자막 이미지 생성
                subtitle_img = create_subtitle_image(
                    paragraph, 
                    image_clip.w, 
                    image_clip.h, 
                    font_path=NANUM_GOTHIC_FONT
                )
                
                # 자막 이미지를 ImageClip으로 변환
                subtitle_clip = ImageClip(subtitle_img, transparent=True)
                subtitle_clip = subtitle_clip.set_duration(audio_clip.duration)
                
                # 모든 클립 합성
                print(f"자막 클립을 비디오에 합성합니다.")
                final_clip = CompositeVideoClip([video_clip, subtitle_clip])
                clips.append(final_clip)
                
            except Exception as e:
                print(f"문단 {i} 처리 중 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=f"비디오 생성 중 오류 발생: {str(e)}")
        
        if not clips:
            raise HTTPException(status_code=500, detail="생성할 비디오 클립이 없습니다.")
        
        # 모든 클립 연결
        final_video = concatenate_videoclips(clips)
        
        # 고유한 파일명 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # 비디오 저장
        try:
            print(f"최종 비디오 저장 시작: {video_path}")
            final_video.write_videofile(
                video_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(temp_dir, 'temp_audio.m4a'),
                remove_temp=True,
                fps=24
            )
            print(f"최종 비디오 저장 완료: {video_path}")
        except Exception as e:
            print(f"비디오 저장 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail=f"비디오 저장 중 오류 발생: {str(e)}")
        
        # 비디오 URL 반환
        return f"{DOMAIN_URL}/videos/{video_filename}"

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