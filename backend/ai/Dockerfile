# 1. 파이썬 설치된 가상 컴퓨터 이미지 가져오기
FROM python:3.11-slim

# 2. 필수 시스템 도구 설치 (ffmpeg 등)
RUN apt-get update && apt-get install -y ffmpeg curl && rm -rf /var/lib/apt/lists/*

# 3. 서버 내 작업 폴더 설정
WORKDIR /app

# 4. 라이브러리 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 내 코드 전체 복사
COPY . .

# 6. 서버 실행 명령어 (8001 포트 사용)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]