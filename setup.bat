@echo off
echo 🚀 SnapSum Backend 개발 환경 자동 설정 시작...

:: Python 버전 확인 및 자동 설치 (Windows에서는 pyenv 없이 직접 설치해야 함)
if exist .python-version (
    set /p PYTHON_VERSION=<.python-version
    echo 📌 프로젝트에서 요구하는 Python 버전: %PYTHON_VERSION%
    echo ⚠️ Windows에서는 pyenv가 없으므로 직접 Python %PYTHON_VERSION%을 설치해야 합니다!
)

:: 가상환경 생성
if not exist "venv" (
    echo 📌 가상환경 생성 중...
    python -m venv venv --prompt SnapSum
)

:: 가상환경 활성화
echo 📌 가상환경 활성화
call venv\Scripts\activate && echo ✅ 가상환경 활성화 완료!

:: pip 최신 버전 업데이트
echo 📌 pip 최신 버전 업데이트
pip install --upgrade pip

:: 패키지 설치
echo 📌 프로젝트 패키지 설치 중...
pip install -r requirements.txt

:: 환경 변수 파일 설정 (.env.example이 있으면 .env 생성)
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo ✅ .env 파일이 생성되었습니다. 필요하면 값을 수정하세요.
    ) else (
        echo ⚠️ .env.example 파일이 없습니다! 수동으로 .env를 생성하세요.
    )
)

echo 🎉 개발 환경 설정 완료! 'python src\main.py' 실행하세요.
pause