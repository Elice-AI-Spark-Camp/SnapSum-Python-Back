import os
import uuid
import aiohttp
import tempfile
import asyncio
import subprocess
import hashlib
import io
import time
from typing import List, Dict
from fastapi import HTTPException
from dotenv import load_dotenv
from services.tts_service import generate_tts
from PIL import Image
import shutil

# .env 로드
load_dotenv()
DOMAIN_URL = os.getenv("DOMAIN_URL", "http://localhost:5001")

# 디렉토리 설정
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 시스템 리소스 확인
CPU_COUNT = os.cpu_count() or 2
print(f"시스템 정보: CPU {CPU_COUNT}코어")

# GPU 확인
HAS_GPU = False
try:
    # nvidia-smi 명령어로 GPU 확인
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        HAS_GPU = True
        print("NVIDIA GPU 감지됨, 하드웨어 가속 사용 가능")
        
        # CUDA 버전 확인
        cuda_info = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"CUDA 정보: {cuda_info.stdout.strip()}")
except:
    print("NVIDIA GPU를 감지할 수 없음, 소프트웨어 인코딩 사용")

# FFmpeg 경로 확인
FFMPEG_PATH = shutil.which("ffmpeg")
if not FFMPEG_PATH:
    # 일반적인 FFmpeg 설치 경로 확인
    possible_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        os.path.expanduser("~/bin/ffmpeg")
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            FFMPEG_PATH = path
            break
            
    if not FFMPEG_PATH:
        raise RuntimeError("FFmpeg가 설치되어 있지 않습니다. FFmpeg를 설치해주세요.")

print(f"FFmpeg 경로: {FFMPEG_PATH}")

# FFmpeg 하드웨어 가속 지원 확인
if HAS_GPU:
    try:
        result = subprocess.run([FFMPEG_PATH, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "h264_nvenc" in result.stdout:
            print("FFmpeg NVENC 하드웨어 가속 지원 확인됨")
            HAS_NVENC = True
        else:
            print("FFmpeg NVENC 하드웨어 가속 지원되지 않음")
            HAS_NVENC = False
    except:
        print("FFmpeg 인코더 확인 실패")
        HAS_NVENC = False
else:
    HAS_NVENC = False

# 최적 동시 작업 수 계산
# A100 GPU는 강력하지만, 메모리 사용량을 고려하여 동시 작업 수 제한
MAX_CONCURRENT_TASKS = min(CPU_COUNT * 2, 8)  # CPU 코어 수의 2배, 최대 8개
print(f"최대 동시 작업 수: {MAX_CONCURRENT_TASKS}")

# 품질 설정 (A100 GPU 최적화)
QUALITY_SETTINGS = {
    "low": {
        "resolution": (640, 360),
        "preset": "p1" if HAS_NVENC else "ultrafast",  # NVENC 프리셋
        "crf": "32" if not HAS_NVENC else None,  # NVENC는 CRF 대신 b:v 사용
        "audio_bitrate": "96k",
        "video_bitrate": "1M" if HAS_NVENC else None
    },
    "medium": {
        "resolution": (854, 480),
        "preset": "p3" if HAS_NVENC else "veryfast",
        "crf": "28" if not HAS_NVENC else None,
        "audio_bitrate": "128k",
        "video_bitrate": "2M" if HAS_NVENC else None
    },
    "high": {
        "resolution": (1280, 720),
        "preset": "p5" if HAS_NVENC else "faster",
        "crf": "23" if not HAS_NVENC else None,
        "audio_bitrate": "192k",
        "video_bitrate": "4M" if HAS_NVENC else None
    }
}

def get_cache_key(paragraph, voice_id, image_url, quality):
    """캐시 키 생성"""
    data = f"{paragraph}|{voice_id}|{image_url}|{quality}"
    return hashlib.md5(data.encode()).hexdigest()

async def download_and_optimize_image(session, url, dest_path, target_resolution=(854, 480)):
    """이미지 다운로드 후 최적화"""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: HTTP {response.status}")
            
            image_data = await response.read()
            
            # 메모리에서 이미지 처리
            img = Image.open(io.BytesIO(image_data))
            
            # 해상도 조정
            img = img.resize(target_resolution, Image.LANCZOS)
            
            # 이미지 최적화
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=70)
            
            # 파일로 저장
            with open(dest_path, 'wb') as f:
                f.write(buffer.getvalue())
            
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

async def process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality="medium"):
    """각 문단을 병렬로 처리 (캐싱 적용)"""
    try:
        print(f"문단 {i} 처리 시작")
        settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
        
        # 캐시 키 생성
        cache_key = get_cache_key(paragraph, voice_id, image_url, quality)
        cache_path = os.path.join(CACHE_DIR, f"segment_{cache_key}.mp4")
        
        # 캐시 확인
        if os.path.exists(cache_path):
            print(f"문단 {i}의 캐시 발견: {cache_path}")
            output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
            shutil.copy(cache_path, output_path)
            return {
                "index": i,
                "segment_path": output_path
            }
        
        # 이미지 다운로드 및 리사이징
        image_path = os.path.join(temp_dir, f"image_{i}.jpg")
        async with aiohttp.ClientSession() as session:
            await download_and_optimize_image(session, image_url, image_path, settings["resolution"])
        
        # TTS 생성
        audio_url = generate_tts(paragraph, "ko-KR", voice_id)
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 세그먼트 비디오 생성
        output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
        
        # FFmpeg 명령어 구성 (GPU 가속 사용)
        if HAS_NVENC:
            cmd = [
                FFMPEG_PATH, "-y",
                "-loop", "1",
                "-i", image_path,
                "-i", audio_path,
                "-c:v", "h264_nvenc",
                "-c:a", "aac",
                "-b:a", settings["audio_bitrate"],
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-preset", settings["preset"],
                "-b:v", settings["video_bitrate"],
                "-rc:v", "vbr_hq",
                "-movflags", "+faststart",
                "-gpu", "0",
                output_path
            ]
        else:
            cmd = [
                FFMPEG_PATH, "-y",
                "-loop", "1",
                "-i", image_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-tune", "stillimage",  # CPU 인코딩에서는 stillimage 튜닝 사용 가능
                "-c:a", "aac",
                "-b:a", settings["audio_bitrate"],
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-preset", settings["preset"],
                "-crf", settings["crf"],
                "-movflags", "+faststart",
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
        
        # 캐시에 저장
        shutil.copy(output_path, cache_path)
        
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
    image_urls: Dict[str, str],
    quality: str = "medium"
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성 (병렬 처리 및 FFmpeg 사용)"""
    start_time = time.time()
    print(f"비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}, 품질={quality}")
    
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
                
            tasks.append(process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality))
        
        print(f"총 {len(tasks)}개의 문단 처리 태스크 생성")
        
        # 동시 실행 작업 수 제한
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        
        async def limited_task(task_func):
            async with semaphore:
                return await task_func
        
        # 제한된 동시성으로 태스크 실행
        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
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
            "-movflags", "+faststart",  # 웹 스트리밍 최적화
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
        
        total_time = time.time() - start_time
        print(f"최종 비디오 생성 완료: {video_path} (총 소요시간: {total_time:.2f}초)")
        
        # 비디오 URL 반환
        return f"{DOMAIN_URL}/videos/{video_filename}"

# 빠른 비디오 생성 함수 (단일 이미지 + 전체 오디오)
async def generate_quick_video(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str],
    quality: str = "medium"
) -> str:
    """빠른 비디오 생성 (단일 이미지 + 전체 오디오)"""
    start_time = time.time()
    print(f"빠른 비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}")
    settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 첫 번째 이미지 사용
        first_image_url = image_urls.get("0")
        if not first_image_url:
            raise HTTPException(status_code=400, detail="이미지 URL이 없습니다.")
        
        # 이미지 다운로드
        image_path = os.path.join(temp_dir, "image.jpg")
        async with aiohttp.ClientSession() as session:
            await download_and_optimize_image(session, first_image_url, image_path, settings["resolution"])
        
        # 모든 문단을 하나의 텍스트로 결합
        full_text = " ".join(paragraphs)
        
        # TTS 생성
        audio_url = generate_tts(full_text, "ko-KR", voice_id)
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, "audio.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어 (GPU 가속 사용)
        if HAS_NVENC:
            cmd = [
                FFMPEG_PATH, "-y",
                "-loop", "1",
                "-i", image_path,
                "-i", audio_path,
                "-c:v", "h264_nvenc",
                "-c:a", "aac",
                "-b:a", settings["audio_bitrate"],
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-preset", settings["preset"],
                "-b:v", settings["video_bitrate"],
                "-rc:v", "vbr_hq",
                "-movflags", "+faststart",
                "-gpu", "0",
                video_path
            ]
        else:
            cmd = [
                FFMPEG_PATH, "-y",
                "-loop", "1",
                "-i", image_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", settings["audio_bitrate"],
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-preset", settings["preset"],
                "-crf", settings["crf"],
                "-movflags", "+faststart",
                video_path
            ]
        
        # FFmpeg 실행
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"비디오 생성 실패: {stderr.decode()}")
        
        total_time = time.time() - start_time
        print(f"빠른 비디오 생성 완료: {video_path} (총 소요시간: {total_time:.2f}초)")
        
        return f"{DOMAIN_URL}/videos/{video_filename}"