# MLOps 시스템을 위한 Dockerfile
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 기본 패키지 먼저 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 필수 의존성 설치
RUN pip install --no-cache-dir \
    requests>=2.31 \
    fastapi>=0.104.1 \
    uvicorn[standard]>=0.24.0 \
    pandas>=2.1 \
    numpy>=1.24 \
    python-dotenv>=1.0 \
    psutil>=5.9

# ML 관련 패키지 설치
RUN pip install --no-cache-dir \
    scikit-learn>=1.3 \
    mlflow>=2.9 \
    wandb>=0.16 \
    joblib>=1.3

# AWS 관련 패키지 설치
RUN pip install --no-cache-dir \
    boto3>=1.28 \
    s3fs>=2023.9

# 워크플로우 패키지 설치
RUN pip install --no-cache-dir \
    prefect>=2.14 \
    schedule>=1.2

# 기타 패키지 설치
RUN pip install --no-cache-dir \
    pyarrow>=14.0 \
    pytz>=2023.3

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py .
COPY app_mlops.py .

# .env 파일을 위한 디렉토리 생성 (마운트 포인트)
RUN mkdir -p /app/config

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# MLflow 저장소 디렉토리 생성
RUN mkdir -p /app/mlflow

# 데이터 디렉토리 생성
RUN mkdir -p /app/data

# 포트 8000 노출 (FastAPI)
EXPOSE 8000

# 포트 5000 노출 (MLflow)
EXPOSE 5000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 기본 명령어 (MLOps 애플리케이션 실행)
CMD ["python", "app_mlops.py"]