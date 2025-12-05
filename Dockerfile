FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /workspace/yolov8_weapon_detecter

# 시스템 의존성 설치 (OpenCV 및 카메라 관련)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 5000

# 애플리케이션 실행
CMD ["python", "app.py"]