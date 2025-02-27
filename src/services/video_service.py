import os
import uuid
import aiohttp
import tempfile
import asyncio
import subprocess
from typing import List, Dict
from datetime import datetime
from fastapi import HTTPException
from dotenv import load_dotenv
from services.tts_service import generate_tts
from PIL import Image
import shutil

# .env 로드
load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

# 비디오 저장 디렉토리
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

# FFmpeg 경로 확인
FFMPEG_PATH = shutil.which("ffmpeg")
if not FFMPEG_PATH:
    raise RuntimeError("FFmpeg가 설치되어 있지 않습니다. FFmpeg를 설치해주세요.")
print(f"FFmpeg 경로: {FFMPEG_PATH}")

async def download_and_resize_image(session, url, dest_path, target_resolution=(1280, 720)):
    """이미지 다운로드 후 최적 크기로 리사이징"""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: HTTP {response.status}")
            
            image_data = await response.read()
            with open(dest_path, 'wb') as f:
                f.write(image_data)
            
            # 이미지 리사이징
            try:
                img = Image.open(dest_path)
                img = img.resize(target_resolution, Image.LANCZOS)
                img.save(dest_path, optimize=True, quality=85)
            except Exception as e:
                print(f"이미지 리사이징 실패: {str(e)}")
            
            return dest_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: {str(e)}")

async def download_file(session, url, dest_path):
    """파일 다운로드"""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"파일 다운로드 실패: HTTP {response.status}")
            
            with open(dest_path, 'wb') as f:
                f.write(await response.read())
            
            return dest_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 다운로드 실패: {str(e)}")

async def process_paragraph(i, paragraph, image_url, temp_dir, voice_id):
    """각 문단을 병렬로 처리"""
    try:
        print(f"문단 {i} 처리 시작")
        
        # 이미지 다운로드 및 리사이징
        image_path = os.path.join(temp_dir, f"image_{i}.jpg")
        async with aiohttp.ClientSession() as session:
            await download_and_resize_image(session, image_url, image_path)
        
        # TTS 생성
        audio_url = generate_tts(paragraph, "ko-KR", voice_id)
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 세그먼트 비디오 생성
        output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
        
        # FFmpeg 명령어 구성
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-preset", "faster",  # 인코딩 속도 향상
            "-threads", "2",      # 스레드 수 제한
            output_path
        ]
        
        # FFmpeg 실행
        print(f"문단 {i} FFmpeg 실행: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"FFmpeg 오류 (문단 {i}): {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"비디오 세그먼트 생성 실패: {stderr.decode()}")
        
        print(f"문단 {i} 처리 완료: {output_path}")
        return {
            "index": i,
            "segment_path": output_path
        }
    except Exception as e:
        print(f"문단 {i} 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비디오 생성 중 오류 발생: {str(e)}")

async def generate_video_with_tts_and_images(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str]
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성 (병렬 처리 및 FFmpeg 사용)"""
    print(f"비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}")
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"임시 디렉토리 생성: {temp_dir}")
        
        # 병렬로 모든 문단 처리
        tasks = []
        for i, paragraph in enumerate(paragraphs):
            idx_str = str(i)
            image_url = image_urls.get(idx_str)
            
            if not image_url:
                print(f"문단 {idx_str}에 대한 이미지 URL이 없습니다.")
                continue
                
            tasks.append(process_paragraph(i, paragraph, image_url, temp_dir, voice_id))
        
        print(f"총 {len(tasks)}개의 문단 처리 태스크 생성")
        
        # 모든 태스크 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 에러 처리
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"문단 처리 중 오류 발생: {errors[0]}")
            raise HTTPException(status_code=500, detail=f"비디오 생성 중 오류 발생: {str(errors[0])}")
        
        # 성공한 결과만 필터링하고 인덱스 순으로 정렬
        valid_results = [r for r in results if not isinstance(r, Exception)]
        valid_results.sort(key=lambda x: x["index"])
        
        if not valid_results:
            print("생성할 비디오 세그먼트가 없습니다.")
            raise HTTPException(status_code=500, detail="생성할 비디오 세그먼트가 없습니다.")
        
        print(f"총 {len(valid_results)}개의 비디오 세그먼트 생성 완료")
        
        # 세그먼트 목록 파일 생성
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for result in valid_results:
                f.write(f"file '{result['segment_path']}'\n")
        
        # 최종 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어로 모든 세그먼트 연결
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",  # 스트림 복사 (재인코딩 없음)
            video_path
        ]
        
        print(f"최종 비디오 생성 명령어: {' '.join(cmd)}")
        
        # FFmpeg 실행
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print(f"최종 비디오 생성 실패: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"최종 비디오 생성 실패: {stderr.decode()}")
        
        print(f"최종 비디오 생성 완료: {video_path}")
        
        # 비디오 URL 반환
        return f"{DOMAIN_URL}/videos/{video_filename}"

def generate_video_file(video_id: int) -> str:
    """간단한 테스트 비디오 생성 함수"""
    # UUID를 사용하여 고유한 파일명 생성
    unique_filename = f"video_{uuid.uuid4()}.mp4"
    video_filename = f"{VIDEO_DIR}/{unique_filename}"

    # 간단한 텍스트 비디오 생성
    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s=1280x720:d=5",
        "-vf", f"drawtext=text='Video ID\\: {video_id}\\nGenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264",
        "-preset", "faster",
        "-pix_fmt", "yuv420p",
        video_filename
    ]
    
    subprocess.run(cmd, check=True)
    
    return f"{DOMAIN_URL}/videos/{unique_filename}"