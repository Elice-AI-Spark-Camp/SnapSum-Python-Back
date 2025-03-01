import os
import uuid
import aiohttp
import tempfile
import asyncio
import subprocess
import hashlib
import io
import time
import logging
from typing import List, Dict
from fastapi import HTTPException
from dotenv import load_dotenv
from services.tts_service import generate_tts
from PIL import Image
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("video_service")

# 상세 디버그 로그 비활성화
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

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
logger.info(f"시스템 정보: CPU {CPU_COUNT}코어")

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

logger.info(f"FFmpeg 경로: {FFMPEG_PATH}")

# NVENC 사용 가능 여부 확인 (더 엄격한 검사)
HAS_NVENC = False
try:
    # 실제 인코딩 테스트로 NVENC 사용 가능 여부 확인
    test_cmd = [
        FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
        "-c:v", "h264_nvenc", "-f", "null", "-"
    ]
    
    result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    
    if result.returncode == 0:
        logger.info("NVENC 하드웨어 가속 사용 가능 (테스트 성공)")
        HAS_NVENC = True
    else:
        logger.info(f"NVENC 하드웨어 가속 사용 불가 (테스트 실패)")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"NVENC 오류: {result.stderr.decode()}")
        HAS_NVENC = False
except Exception as e:
    logger.info(f"NVENC 테스트 중 오류 발생")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"상세 오류: {str(e)}")
    HAS_NVENC = False

# 최적 동시 작업 수 증가
MAX_CONCURRENT_TASKS = min(CPU_COUNT * 2, 8)  # CPU 코어 수의 2배, 최대 8개

# 품질 설정 최적화 (인코딩 속도 향상)
QUALITY_SETTINGS = {
    "low": {
        "resolution": (720, 1280),  # 세로 비디오 (9:16 비율)
        "preset": "veryfast",  # 더 빠른 인코딩
        "crf": "28",
        "audio_bitrate": "64k"
    },
    "medium": {
        "resolution": (1080, 1920),  # 세로 비디오 (9:16 비율)
        "preset": "veryfast",  # 더 빠른 인코딩
        "crf": "26",  # 약간 낮은 품질, 더 빠른 인코딩
        "audio_bitrate": "128k"
    },
    "high": {
        "resolution": (1440, 2560),  # 세로 비디오 (9:16 비율)
        "preset": "fast",  # 더 빠른 인코딩
        "crf": "23",
        "audio_bitrate": "192k"
    }
}

# 캐시 키 생성 함수 최적화
def get_cache_key(paragraph, voice_id, image_url, quality):
    """캐시 키 생성 (해시 기반)"""
    # 입력 데이터를 결합하여 해시 생성
    combined = f"{paragraph}|{voice_id}|{image_url}|{quality}"
    return hashlib.md5(combined.encode()).hexdigest()

# 이미지 최적화 간소화
async def download_and_optimize_image(session, url, output_path, target_resolution):
    """이미지 다운로드 및 최적화 (간소화)"""
    try:
        # 이미지 다운로드
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: {url}, 상태 코드: {response.status}")
            
            image_data = await response.read()
            
            # 이미지 처리 (간소화)
            with Image.open(io.BytesIO(image_data)) as img:
                # 리사이징만 수행 (복잡한 처리 제거)
                target_width, target_height = target_resolution
                
                # 세로 비디오에 맞게 이미지 조정
                img_width, img_height = img.size
                ratio = min(target_width / img_width, target_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 리사이징 및 중앙 정렬
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                new_img.paste(resized_img, (paste_x, paste_y))
                
                # 최적화된 품질로 저장
                new_img.save(output_path, format="JPEG", quality=85, optimize=True)
                
            return output_path
    except Exception as e:
        logger.error(f"이미지 다운로드 및 최적화 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 실패: {str(e)}")

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

# 동기 TTS 래퍼 함수
def sync_generate_tts(text, language_code, voice_id):
    """TTS 생성 동기 래퍼 함수"""
    from services.tts_service import generate_tts
    return generate_tts(text, language_code, voice_id)

# 문단 처리 함수 최적화
async def process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality="medium"):
    """각 문단을 병렬로 처리 (캐싱 적용)"""
    try:
        if i % 3 == 0:  # 로그 줄이기
            logger.info(f"문단 {i} 처리 시작")
        settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
        
        # 캐시 키 생성
        cache_key = get_cache_key(paragraph, voice_id, image_url, quality)
        cache_path = os.path.join(CACHE_DIR, f"segment_{cache_key}.mp4")
        
        # 캐시 확인
        if os.path.exists(cache_path):
            if i % 3 == 0:
                logger.info(f"문단 {i}의 캐시 발견: 처리 시간 단축")
            output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
            shutil.copy(cache_path, output_path)
            return {
                "index": i,
                "segment_path": output_path
            }
        
        # 이미지와 TTS 병렬 처리
        async with aiohttp.ClientSession() as session:
            # 이미지와 TTS 동시에 처리
            image_task = asyncio.create_task(
                download_and_optimize_image(session, image_url, os.path.join(temp_dir, f"image_{i}.jpg"), settings["resolution"])
            )
            
            # TTS 생성 (동기 함수를 별도 스레드에서 실행)
            loop = asyncio.get_event_loop()
            tts_task = loop.run_in_executor(
                None,  # 기본 executor 사용
                lambda: sync_generate_tts(paragraph, "ko-KR", voice_id)
            )
            
            # 두 작업 동시에 기다림
            image_path, audio_url = await asyncio.gather(image_task, tts_task)
            
            # 오디오 다운로드
            audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
            await download_file(session, audio_url, audio_path)
        
        # 세그먼트 비디오 생성
        output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
        
        # FFmpeg 명령어 구성 (최적화된 설정)
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264" if not HAS_NVENC else "h264_nvenc",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-preset", settings["preset"] if not HAS_NVENC else "p1",
            "-crf", settings["crf"] if not HAS_NVENC else "auto",
            # 세로 비디오 설정
            "-vf", f"scale={settings['resolution'][0]}:{settings['resolution'][1]}:force_original_aspect_ratio=decrease,pad={settings['resolution'][0]}:{settings['resolution'][1]}:(ow-iw)/2:(oh-ih)/2",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path
        ]
        
        # FFmpeg 실행
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"FFmpeg 오류 (문단 {i}): {error_msg[:200]}...")
            raise HTTPException(status_code=500, detail=f"비디오 세그먼트 생성 실패: {error_msg}")
        
        # 캐시에 저장
        shutil.copy(output_path, cache_path)
        
        if i % 3 == 0:
            logger.info(f"문단 {i} 처리 완료 및 캐시 저장")
        return {
            "index": i,
            "segment_path": output_path
        }
    except Exception as e:
        logger.error(f"문단 {i} 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비디오 생성 중 오류 발생: {str(e)}")

# 비디오 생성 함수 최적화 (배치 처리)
async def generate_video_with_tts_and_images(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str],
    quality: str = "medium"
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성 (배치 처리 및 캐싱 최적화)"""
    start_time = time.time()
    logger.info(f"비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}, 품질={quality}")
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 배치 처리 (메모리 효율성 개선)
        batch_size = min(MAX_CONCURRENT_TASKS, 4)  # 배치 크기 제한
        all_segments = []
        
        for batch_start in range(0, len(paragraphs), batch_size):
            batch_end = min(batch_start + batch_size, len(paragraphs))
            logger.info(f"배치 처리: {batch_start}~{batch_end-1} ({batch_end-batch_start}개 문단)")
            
            batch_tasks = []
            for i in range(batch_start, batch_end):
                paragraph = paragraphs[i]
                idx_str = str(i)
                image_url = image_urls.get(idx_str)
                
                if not image_url:
                    logger.warning(f"문단 {i}의 이미지 URL이 없습니다. 기본 이미지 사용")
                    image_url = image_urls.get("0") or list(image_urls.values())[0]
                
                batch_tasks.append(process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality))
            
            # 배치 실행
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 오류 확인
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"문단 처리 중 오류 발생: {str(result)}")
                    raise result
                all_segments.append(result)
            
            # 메모리 정리
            import gc
            gc.collect()
        
        # 세그먼트 정렬 및 병합
        all_segments.sort(key=lambda x: x["index"])
        segment_paths = [segment["segment_path"] for segment in all_segments]
        
        # 세그먼트 목록 파일 생성
        concat_file_path = os.path.join(temp_dir, "concat.txt")
        with open(concat_file_path, 'w') as f:
            for segment_path in segment_paths:
                f.write(f"file '{segment_path}'\n")
        
        # 최종 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어 구성
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",
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
            error_msg = stderr.decode()
            logger.error(f"최종 비디오 생성 실패: {error_msg[:200]}...")
            raise HTTPException(status_code=500, detail=f"최종 비디오 생성 실패: {error_msg}")
        
        # 처리 시간 로깅
        elapsed_time = time.time() - start_time
        logger.info(f"비디오 생성 완료: {video_path}, 소요 시간: {elapsed_time:.2f}초")
        
        return f"{DOMAIN_URL}/videos/{video_filename}"

# 빠른 비디오 생성 함수 최적화
async def generate_quick_video(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str],
    quality: str = "medium"
) -> str:
    """빠른 비디오 생성 (단일 이미지 + 전체 오디오) - 최적화"""
    start_time = time.time()
    logger.info(f"빠른 비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}")
    settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
    
    # 캐시 키 생성 (전체 텍스트 기반)
    full_text = " ".join(paragraphs)
    first_image_url = image_urls.get("0") or list(image_urls.values())[0]
    cache_key = get_cache_key(full_text, voice_id, first_image_url, quality)
    cache_path = os.path.join(CACHE_DIR, f"quick_video_{cache_key}.mp4")
    
    # 캐시 확인
    if os.path.exists(cache_path):
        logger.info(f"빠른 비디오 캐시 발견: {cache_path}")
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        shutil.copy(cache_path, video_path)
        logger.info(f"캐시된 비디오 복사 완료: {video_path}")
        return f"{DOMAIN_URL}/videos/{video_filename}"
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 이미지 다운로드
        image_path = os.path.join(temp_dir, "image.jpg")
        async with aiohttp.ClientSession() as session:
            await download_and_optimize_image(session, first_image_url, image_path, settings["resolution"])
        
        # TTS 생성 (별도 스레드에서 실행)
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(
            None,
            lambda: sync_generate_tts(full_text, "ko-KR", voice_id)
        )
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, "audio.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어 (최적화된 설정)
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264" if not HAS_NVENC else "h264_nvenc",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-preset", settings["preset"] if not HAS_NVENC else "p1",
            "-crf", settings["crf"] if not HAS_NVENC else "auto",
            # 세로 비디오 설정
            "-vf", f"scale={settings['resolution'][0]}:{settings['resolution'][1]}:force_original_aspect_ratio=decrease,pad={settings['resolution'][0]}:{settings['resolution'][1]}:(ow-iw)/2:(oh-ih)/2",
            "-movflags", "+faststart",
            "-loglevel", "error",
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
            error_msg = stderr.decode()
            logger.error(f"빠른 비디오 생성 실패: {error_msg[:200]}...")
            raise HTTPException(status_code=500, detail=f"비디오 생성 실패: {error_msg}")
        
        # 캐시에 저장
        shutil.copy(video_path, cache_path)
        
        total_time = time.time() - start_time
        logger.info(f"빠른 비디오 생성 완료: 총 소요시간: {total_time:.2f}초")
        
        return f"{DOMAIN_URL}/videos/{video_filename}"
        