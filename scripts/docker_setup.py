#!/usr/bin/env python3
"""도커 환경 설정 및 실행 스크립트."""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """명령어 실행 및 결과 출력."""
    print(f"🔄 {description}")
    print(f"실행: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ 출력:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류: {e}")
        if e.stderr:
            print(f"에러 메시지:\n{e.stderr}")
        return False

def check_docker():
    """도커 설치 및 실행 상태 확인."""
    print("🐳 도커 환경 확인 중...")

    # 도커 설치 확인
    if not run_command("docker --version", "도커 버전 확인"):
        print("❌ 도커가 설치되지 않았습니다.")
        print("도커 설치: https://docs.docker.com/get-docker/")
        return False

    # 도커 데몬 실행 확인
    if not run_command("docker info > /dev/null 2>&1", "도커 데몬 상태 확인"):
        print("❌ 도커 데몬이 실행되지 않았습니다.")
        print("Docker Desktop을 실행해주세요.")
        return False

    # Docker Compose 확인
    if not run_command("docker compose version", "Docker Compose 버전 확인"):
        print("❌ Docker Compose가 설치되지 않았습니다.")
        return False

    print("✅ 도커 환경 확인 완료")
    return True

def check_env_file():
    """환경 변수 파일 확인."""
    print("🔧 환경 변수 파일 확인 중...")

    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env 파일이 없습니다.")
        print("환경 변수 파일을 생성하겠습니다.")

        # 기본 .env 파일 생성
        env_content = """# 기상청 API 키 (필수)
KMA_AUTH_KEY=your_kma_api_key_here

# WandB API 키 (실험 추적용)
WANDB_API_KEY=your_wandb_api_key_here

# AWS S3 설정 (선택사항)
COMMUTE_S3_BUCKET=your-s3-bucket-name
AWS_DEFAULT_REGION=ap-northeast-2

# MLflow 설정
MLFLOW_TRACKING_URI=file:///app/mlflow

# WandB 프로젝트 설정
WANDB_PROJECT=commute-weather-mlops

# 기상청 관측소 ID (서울: 108)
KMA_STATION_ID=108

# 애플리케이션 설정
PYTHONPATH=/app/src
"""

        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print(f"✅ .env 파일 생성: {env_file.absolute()}")
        print("🚨 중요: .env 파일에서 API 키를 설정해주세요!")
        return False

    print("✅ .env 파일 확인 완료")
    return True

def build_images():
    """도커 이미지 빌드."""
    print("🔨 도커 이미지 빌드 중...")

    return run_command(
        "docker compose build --no-cache",
        "MLOps 도커 이미지 빌드"
    )

def start_services():
    """서비스 시작."""
    print("🚀 MLOps 서비스 시작 중...")

    return run_command(
        "docker compose up -d",
        "MLOps 서비스 시작"
    )

def stop_services():
    """서비스 중지."""
    print("⏹️ MLOps 서비스 중지 중...")

    return run_command(
        "docker compose down",
        "MLOps 서비스 중지"
    )

def show_status():
    """서비스 상태 확인."""
    print("📊 서비스 상태 확인 중...")

    run_command("docker compose ps", "실행 중인 컨테이너")
    print("\n📋 서비스 엔드포인트:")
    print("- MLOps 애플리케이션: http://localhost:8000")
    print("- MLOps 대시보드: http://localhost:8000/mlops/health")
    print("- MLflow 서버: http://localhost:5001")
    print("- API 문서: http://localhost:8000/docs")

def show_logs():
    """서비스 로그 확인."""
    print("📋 서비스 로그:")

    run_command("docker compose logs --tail=50", "최근 로그 50줄")

def test_feature_engineering():
    """피처 엔지니어링 테스트."""
    print("🧪 피처 엔지니어링 테스트 실행 중...")

    return run_command(
        "docker compose exec mlops-app python scripts/test_feature_engineering.py",
        "피처 엔지니어링 테스트"
    )

def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="MLOps 도커 환경 관리")
    parser.add_argument("action", choices=[
        "setup", "build", "start", "stop", "restart",
        "status", "logs", "test", "clean"
    ], help="실행할 작업")

    args = parser.parse_args()

    print("🐳 MLOps 도커 환경 관리")
    print("=" * 50)

    if args.action == "setup":
        print("🔧 초기 설정 시작...")

        if not check_docker():
            sys.exit(1)

        if not check_env_file():
            print("\n⚠️ .env 파일을 수정한 후 다시 실행해주세요.")
            sys.exit(1)

        if build_images():
            print("✅ 도커 환경 설정 완료!")
            print("\n다음 명령어로 서비스를 시작하세요:")
            print("python scripts/docker_setup.py start")
        else:
            print("❌ 도커 이미지 빌드 실패")
            sys.exit(1)

    elif args.action == "build":
        if build_images():
            print("✅ 도커 이미지 빌드 완료!")
        else:
            sys.exit(1)

    elif args.action == "start":
        if start_services():
            print("✅ MLOps 서비스 시작 완료!")
            show_status()
        else:
            sys.exit(1)

    elif args.action == "stop":
        if stop_services():
            print("✅ MLOps 서비스 중지 완료!")
        else:
            sys.exit(1)

    elif args.action == "restart":
        print("🔄 서비스 재시작 중...")
        stop_services()
        if start_services():
            print("✅ MLOps 서비스 재시작 완료!")
            show_status()
        else:
            sys.exit(1)

    elif args.action == "status":
        show_status()

    elif args.action == "logs":
        show_logs()

    elif args.action == "test":
        if test_feature_engineering():
            print("✅ 테스트 완료!")
        else:
            print("❌ 테스트 실패")
            sys.exit(1)

    elif args.action == "clean":
        print("🧹 정리 중...")
        stop_services()
        run_command("docker compose down -v", "볼륨 정리")
        run_command("docker system prune -f", "사용하지 않는 도커 리소스 정리")
        print("✅ 정리 완료!")

if __name__ == "__main__":
    main()