# 베이스 이미지로 Python 3.10 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 실행 명령어 추가
CMD ["python", "server.py"]

# 컨테이너 실행 시 스크립트가 데이터 폴더에 접근할 수 있도록 설정
# (실제 데이터 마운트는 docker-compose에서 처리)
