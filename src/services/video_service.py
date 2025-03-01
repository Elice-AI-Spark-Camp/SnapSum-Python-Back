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
HAS_GPU = False

try:
    # A100 GPU 확인 (CUDA 라이브러리 사용)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU 감지됨: {gpu_name}")
            HAS_GPU = True
            if "A100" in gpu_name:
                logger.info("A100 GPU 감지됨")
    except ImportError:
        logger.info("PyTorch가 설치되지 않아 GPU 확인 불가")
    
    # NVENC 테스트
    if HAS_GPU:
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
except Exception as e:
    logger.info(f"GPU 테스트 중 오류 발생: {str(e)}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"상세 오류: {str(e)}")

# 최적 동시 작업 수 계산 (GPU 있을 경우 더 많은 병렬 처리)
MAX_CONCURRENT_TASKS = min(CPU_COUNT * 2, 16) if HAS_GPU else min(CPU_COUNT * 2, 8)
logger.info(f"최대 동시 작업 수: {MAX_CONCURRENT_TASKS}")

# 품질 설정 (GPU 최적화 포함)
QUALITY_SETTINGS = {
    "low": {
        "resolution": (480, 854),  # 세로형 비디오 (9:16 비율)
        "audio_bitrate": "64k",
        "preset": "ultrafast",
        "crf": "28",
        "gpu_options": ["-hwaccel", "cuda"] if HAS_GPU else []
    },
    "medium": {
        "resolution": (720, 1280),  # 세로형 비디오 (9:16 비율)
        "audio_bitrate": "128k",
        "preset": "medium",
        "crf": "23",
        "gpu_options": ["-hwaccel", "cuda"] if HAS_GPU else []
    },
    "high": {
        "resolution": (1080, 1920),  # 세로형 비디오 (9:16 비율)
        "audio_bitrate": "192k",
        "preset": "slow",
        "crf": "18",
        "gpu_options": ["-hwaccel", "cuda"] if HAS_GPU else []
    }
}

def get_cache_key(paragraph, voice_id, image_url, quality):
    """캐시 키 생성"""
    data = f"{paragraph}|{voice_id}|{image_url}|{quality}"
    return hashlib.md5(data.encode()).hexdigest()

async def download_and_optimize_image(session, url, output_path, target_resolution=(720, 1280)):
    """이미지 다운로드 및 최적화"""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"이미지 다운로드 실패: HTTP {response.status}")
            
            image_data = await response.read()
            
            # 이미지 처리
            with Image.open(io.BytesIO(image_data)) as img:
                # 원본 이미지 비율 계산
                width, height = img.size
                aspect_ratio = width / height
                
                # 세로형 비디오에 맞게 이미지 조정
                target_width, target_height = target_resolution
                target_aspect = target_width / target_height  # 9:16 비율
                
                if aspect_ratio > target_aspect:  # 원본이 더 가로로 넓은 경우
                    # 세로 크기에 맞추고 가로는 크롭
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 중앙 크롭
                    left = (new_width - target_width) // 2
                    right = left + target_width
                    cropped = resized.crop((left, 0, right, new_height))
                else:  # 원본이 더 세로로 긴 경우
                    # 가로 크기에 맞추고 세로는 크롭 또는 패딩
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
                    
                    if new_height >= target_height:  # 크롭 필요
                        resized = img.resize((new_width, new_height), Image.LANCZOS)
                        top = (new_height - target_height) // 2
                        bottom = top + target_height
                        cropped = resized.crop((0, top, new_width, bottom))
                    else:  # 패딩 필요
                        # 검은색 배경 생성
                        background = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                        resized = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # 중앙에 배치
                        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
                        background.paste(resized, offset)
                        cropped = background
                
                # 최적화된 이미지 저장
                cropped.save(output_path, format="JPEG", quality=90, optimize=True)
                
                return output_path
    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류: {str(e)}")

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

async def process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality="medium", use_gpu=False):
    """각 문단을 병렬로 처리 (캐싱 적용)"""
    try:
        if i % 5 == 0:  # 5개 문단마다 로그 출력
            logger.info(f"문단 {i} 처리 시작")
        settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
        
        # 캐시 키 생성
        cache_key = get_cache_key(paragraph, voice_id, image_url, quality)
        cache_path = os.path.join(CACHE_DIR, f"segment_{cache_key}.mp4")
        
        # 캐시 확인
        if os.path.exists(cache_path):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"문단 {i}의 캐시 발견: {cache_path}")
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
        
        # TTS 생성 (비동기 함수로 변경되었으므로 await 사용)
        audio_url = await generate_tts(paragraph, "ko-KR", voice_id)
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, f"audio_{i}.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 세그먼트 비디오 생성
        output_path = os.path.join(temp_dir, f"segment_{i}.mp4")
        
        # FFmpeg 명령어 구성
        cmd = [FFMPEG_PATH, "-y"]
        
        # GPU 하드웨어 가속 옵션 추가
        if use_gpu and HAS_GPU:
            cmd.extend(settings["gpu_options"])
        
        # 입력 파일 추가
        cmd.extend([
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path
        ])
        
        # 인코딩 설정
        if use_gpu and HAS_NVENC:
            # NVENC 하드웨어 인코딩 사용
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-tune", "stillimage",
                "-preset", "p1"  # NVENC 프리셋
            ])
        else:
            # CPU 인코딩 또는 GPU 가속 + CPU 인코딩
            cmd.extend([
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-preset", settings["preset"],
                "-crf", settings["crf"]
            ])
        
        # 공통 설정
        cmd.extend([
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path
        ])
        
        # FFmpeg 실행
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"문단 {i} FFmpeg 실행: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"FFmpeg 오류 (문단 {i}): {error_msg[:200]}...")  # 오류 메시지 일부만 로깅
            raise HTTPException(status_code=500, detail=f"비디오 세그먼트 생성 실패: {error_msg}")
        
        # 캐시에 저장
        shutil.copy(output_path, cache_path)
        
        if i % 5 == 0:  # 5개 문단마다 로그 출력
            logger.info(f"문단 {i} 처리 완료")
        return {
            "index": i,
            "segment_path": output_path
        }
    except Exception as e:
        logger.error(f"문단 {i} 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비디오 생성 중 오류 발생: {str(e)}")

async def generate_video_with_tts_and_images(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str],
    quality: str = "medium",
    use_gpu: bool = False,
    batch_size: int = None
) -> str:
    """TTS와 이미지를 사용하여 비디오 생성 (병렬 처리 및 FFmpeg 사용)"""
    start_time = time.time()
    logger.info(f"비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}, 품질={quality}, GPU 사용={use_gpu}")
    
    # 동시 작업 수 설정 (batch_size가 지정되면 해당 값 사용)
    concurrent_tasks = batch_size if batch_size else MAX_CONCURRENT_TASKS
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"임시 디렉토리 생성: {temp_dir}")
        
        # 병렬로 모든 문단 처리
        tasks = []
        for i, paragraph in enumerate(paragraphs):
            idx_str = str(i)
            image_url = image_urls.get(idx_str)
            
            if not image_url:
                logger.warning(f"문단 {i}의 이미지 URL이 없습니다. 기본 이미지 사용")
                # 기본 이미지 URL 설정 (필요시)
                image_url = image_urls.get("0") or list(image_urls.values())[0]
            
            tasks.append(process_paragraph(i, paragraph, image_url, temp_dir, voice_id, quality, use_gpu))
        
        logger.info(f"총 {len(tasks)}개의 문단 처리 태스크 생성, 동시 처리 수: {concurrent_tasks}")
        
        # 세마포어를 사용하여 동시 작업 수 제한
        semaphore = asyncio.Semaphore(concurrent_tasks)
        
        async def limited_task(task_func):
            async with semaphore:
                return await task_func
        
        # 제한된 동시성으로 태스크 실행
        limited_tasks = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # 오류 확인
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.error(f"{len(errors)}개의 문단 처리 중 오류 발생")
            raise errors[0]
        
        # 성공한 결과만 필터링
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # 인덱스 기준으로 정렬
        sorted_results = sorted(successful_results, key=lambda x: x["index"])
        
        # 세그먼트 파일 목록 생성
        segment_files = [result["segment_path"] for result in sorted_results]
        
        if not segment_files:
            logger.error("처리된 세그먼트가 없습니다.")
            raise HTTPException(status_code=500, detail="비디오 세그먼트 생성 실패")
        
        # 세그먼트 파일 목록 파일 생성
        segments_list_path = os.path.join(temp_dir, "segments.txt")
        with open(segments_list_path, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{segment_file}'\n")
        
        # 최종 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어 (세그먼트 연결)
        cmd = [FFMPEG_PATH, "-y"]
        
        # GPU 하드웨어 가속 옵션 추가 (최종 연결에도 GPU 사용)
        if use_gpu and HAS_GPU:
            settings = QUALITY_SETTINGS.get(quality, QUALITY_SETTINGS["medium"])
            cmd.extend(settings["gpu_options"])
        
        cmd.extend([
            "-f", "concat",
            "-safe", "0",
            "-i", segments_list_path,
            "-c", "copy",  # 단순 연결이므로 copy 사용
            "-movflags", "+faststart",
            "-loglevel", "error",  # FFmpeg 로그 레벨 조정
            video_path
        ])
        
        logger.info("최종 비디오 생성 시작")
        
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
        
        total_time = time.time() - start_time
        logger.info(f"최종 비디오 생성 완료: 총 소요시간: {total_time:.2f}초")
        
        # 비디오 URL 반환
        return f"{DOMAIN_URL}/videos/{video_filename}"

# 빠른 비디오 생성 함수 (단일 이미지 + 전체 오디오)
async def generate_quick_video(
    summary_id: int,
    paragraphs: List[str],
    voice_id: str,
    image_urls: Dict[str, str],
    quality: str = "medium",
    use_gpu: bool = False
) -> str:
    """빠른 비디오 생성 (단일 이미지 + 전체 오디오)"""
    start_time = time.time()
    logger.info(f"빠른 비디오 생성 시작: summary_id={summary_id}, 문단 수={len(paragraphs)}, GPU 사용={use_gpu}")
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
        audio_url = await generate_tts(full_text, "ko-KR", voice_id)
        
        # 오디오 다운로드
        audio_path = os.path.join(temp_dir, "audio.wav")
        async with aiohttp.ClientSession() as session:
            await download_file(session, audio_url, audio_path)
        
        # 비디오 생성
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        # FFmpeg 명령어 구성
        cmd = [FFMPEG_PATH, "-y"]
        
        # GPU 하드웨어 가속 옵션 추가
        if use_gpu and HAS_GPU:
            cmd.extend(settings["gpu_options"])
        
        # 입력 파일 추가
        cmd.extend([
            "-loop", "1",
            "-i", image_path,
            "-i", audio_path
        ])
        
        # 인코딩 설정
        if use_gpu and HAS_NVENC:
            # NVENC 하드웨어 인코딩 사용
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-tune", "stillimage",
                "-preset", "p1"  # NVENC 프리셋
            ])
        else:
            # CPU 인코딩 또는 GPU 가속 + CPU 인코딩
            cmd.extend([
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-preset", settings["preset"],
                "-crf", settings["crf"]
            ])
        
        # 공통 설정
        cmd.extend([
            "-c:a", "aac",
            "-b:a", settings["audio_bitrate"],
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            "-loglevel", "error",
            video_path
        ])
        
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
        
        total_time = time.time() - start_time
        logger.info(f"빠른 비디오 생성 완료: 총 소요시간: {total_time:.2f}초")
        
        return f"{DOMAIN_URL}/videos/{video_filename}"