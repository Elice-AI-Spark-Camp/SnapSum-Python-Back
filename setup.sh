#!/bin/bash

echo "🚀 SnapSum Backend 개발 환경 자동 설정 시작..."

# Python 버전 확인 및 자동 설치
if [ -f ".python-version" ]; then
    echo "📌 Python 버전 통일 중..."
    pyenv install -s $(cat .python-version)
    pyenv local $(cat .python-version)
fi

# 가상환경 생성 및 프로젝트 이름으로 프롬프트 설정
if [ ! -d "venv" ]; then
    echo "📌 가상환경 생성 중..."
    python -m venv venv --prompt SnapSum
fi

# 가상환경 활성화
echo "📌 가상환경 활성화"
source venv/bin/activate

# pip 최신 버전 업데이트
pip install --upgrade pip

# 패키지 설치
pip install -r requirements.txt

# 환경 변수 파일 설정 (.env.example이 있으면 .env 생성)
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ .env 파일이 생성되었습니다. 필요하면 값을 수정하세요."
    else
        echo "⚠️ .env.example 파일이 없습니다! 수동으로 .env를 생성하세요."
    fi
fi

echo "🎉 개발 환경 설정 완료! 'python src/main.py' 실행하세요."