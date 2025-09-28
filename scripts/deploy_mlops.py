#!/usr/bin/env python3
"""MLOps 시스템 배포 스크립트."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_environment():
    """환경 변수 및 의존성 확인."""
    print("🔍 환경 설정 확인 중...")

    required_env_vars = [
        "KMA_AUTH_KEY",
        "WANDB_API_KEY"
    ]

    optional_env_vars = [
        "COMMUTE_S3_BUCKET",
        "MLFLOW_TRACKING_URI",
        "WANDB_PROJECT",
        "AWS_DEFAULT_REGION"
    ]

    missing_required = []

    for var in required_env_vars:
        if not os.getenv(var):
            missing_required.append(var)

    if missing_required:
        print("❌ 필수 환경변수가 설정되지 않았습니다:")
        for var in missing_required:
            print(f"   - {var}")
        print("\n📝 .env 파일에 다음 변수들을 설정하세요:")
        print("   KMA_AUTH_KEY=your_kma_api_key_here")
        print("   WANDB_API_KEY=your_wandb_api_key_here")
        return False

    print("✅ 필수 환경변수 설정 완료")

    # 선택적 환경변수 확인
    print("\n📊 선택적 환경변수 상태:")
    for var in optional_env_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            print(f"   ⚠️  {var}: 설정되지 않음")

    return True


def check_dependencies():
    """Python 의존성 확인."""
    print("\n🔍 Python 의존성 확인 중...")

    required_packages = [
        "fastapi",
        "uvicorn",
        "requests",
        "pandas",
        "scikit-learn",
        "mlflow",
        "wandb",
        "boto3",
        "s3fs",
        "prefect",
        "python-dotenv"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")

    if missing_packages:
        print(f"\n❌ 누락된 패키지들: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ 모든 의존성 설치 완료")
    return True


def setup_mlflow():
    """MLflow 설정."""
    print("\n🚀 MLflow 설정 중...")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        # 로컬 MLflow 서버 설정
        mlflow_dir = project_root / "mlflow"
        mlflow_dir.mkdir(exist_ok=True)

        mlflow_uri = f"file://{mlflow_dir.absolute()}"

        # .env 파일에 추가
        env_file = project_root / ".env"
        with open(env_file, "a") as f:
            f.write(f"\nMLFLOW_TRACKING_URI={mlflow_uri}\n")

        print(f"   📁 로컬 MLflow 디렉토리 생성: {mlflow_dir}")

    print(f"   ✅ MLflow URI: {mlflow_uri}")

    # MLflow 서버 시작 (백그라운드)
    if "file://" in mlflow_uri:
        try:
            # MLflow UI 서버가 이미 실행 중인지 확인
            import requests
            requests.get("http://localhost:5000", timeout=1)
            print("   ✅ MLflow UI가 이미 실행 중입니다 (http://localhost:5000)")
        except:
            print("   🚀 MLflow UI 서버 시작 중...")
            mlflow_cmd = [
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", mlflow_uri.replace("file://", ""),
                "--host", "0.0.0.0",
                "--port", "5000"
            ]

            subprocess.Popen(
                mlflow_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("   ✅ MLflow UI 시작됨 (http://localhost:5000)")


def setup_s3():
    """S3 설정 확인."""
    print("\n☁️  S3 설정 확인 중...")

    bucket_name = os.getenv("COMMUTE_S3_BUCKET")
    if not bucket_name:
        print("   ⚠️  S3 버킷이 설정되지 않았습니다.")
        print("   💡 로컬 저장소를 사용합니다.")
        return

    try:
        import boto3
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"   ✅ S3 버킷 연결 확인: {bucket_name}")
    except Exception as e:
        print(f"   ❌ S3 연결 실패: {e}")
        print("   💡 AWS 자격증명을 확인하세요.")


def setup_wandb():
    """WandB 설정 확인."""
    print("\n📊 WandB 설정 확인 중...")

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("   ❌ WANDB_API_KEY가 설정되지 않았습니다.")
        return False

    try:
        import wandb

        # 로그인 테스트
        if wandb.api.api_key:
            print("   ✅ WandB 로그인 상태 확인됨")
        else:
            # API 키로 로그인 시도
            os.environ["WANDB_API_KEY"] = api_key
            wandb.login()
            print("   ✅ WandB 로그인 성공")

        # 프로젝트 설정
        project_name = os.getenv("WANDB_PROJECT", "commute-weather-mlops")
        print(f"   📁 WandB 프로젝트: {project_name}")

        return True

    except Exception as e:
        print(f"   ❌ WandB 설정 실패: {e}")
        return False


def run_mlops_app(port=8000, dev_mode=False):
    """MLOps FastAPI 애플리케이션 실행."""
    print(f"\n🚀 MLOps 애플리케이션 시작 중... (포트: {port})")

    app_file = "app_mlops.py"

    if not Path(app_file).exists():
        print(f"❌ {app_file} 파일을 찾을 수 없습니다.")
        return False

    try:
        if dev_mode:
            # 개발 모드: 자동 재시작
            cmd = [
                sys.executable, "-m", "uvicorn",
                "app_mlops:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload"
            ]
        else:
            # 프로덕션 모드
            cmd = [
                sys.executable, "-m", "uvicorn",
                "app_mlops:app",
                "--host", "0.0.0.0",
                "--port", str(port)
            ]

        print(f"   📱 웹 애플리케이션: http://localhost:{port}")
        print(f"   📊 MLOps 대시보드: http://localhost:{port}/mlops/health")
        print(f"   📈 MLflow UI: http://localhost:5000")
        print("\n🔄 애플리케이션 실행 중... (Ctrl+C로 종료)")

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 애플리케이션 종료됨")
        return True
    except Exception as e:
        print(f"❌ 애플리케이션 실행 실패: {e}")
        return False


def run_pipeline_test():
    """파이프라인 테스트 실행."""
    print("\n🧪 MLOps 파이프라인 테스트 중...")

    try:
        from commute_weather.pipelines.realtime_mlops_pipeline import realtime_mlops_flow

        # 테스트 실행
        result = realtime_mlops_flow(
            auth_key=os.getenv("KMA_AUTH_KEY"),
            station_id="108",
            use_s3=bool(os.getenv("COMMUTE_S3_BUCKET")),
            s3_bucket=os.getenv("COMMUTE_S3_BUCKET"),
            use_wandb=bool(os.getenv("WANDB_API_KEY")),
            wandb_project=os.getenv("WANDB_PROJECT", "commute-weather-mlops")
        )

        if result["status"] == "success":
            print("✅ 파이프라인 테스트 성공!")
            print(f"   📊 예측값: {result.get('prediction_value', 'N/A')}")
            print(f"   ⏱️  실행시간: {result.get('execution_time_seconds', 0):.2f}초")
        else:
            print(f"❌ 파이프라인 테스트 실패: {result.get('error', '알 수 없는 오류')}")

        return result["status"] == "success"

    except Exception as e:
        print(f"❌ 파이프라인 테스트 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MLOps 시스템 배포 스크립트")

    parser.add_argument("command", choices=["setup", "test", "run", "deploy"], help="실행할 명령")
    parser.add_argument("--port", type=int, default=8000, help="애플리케이션 포트 (기본: 8000)")
    parser.add_argument("--dev", action="store_true", help="개발 모드로 실행 (자동 재시작)")
    parser.add_argument("--skip-checks", action="store_true", help="환경 확인 건너뛰기")

    args = parser.parse_args()

    print("🤖 MLOps 시스템 배포 도구")
    print("=" * 50)

    if args.command == "setup":
        print("🔧 MLOps 환경 설정 중...")

        if not args.skip_checks:
            if not check_environment():
                return 1
            if not check_dependencies():
                return 1

        setup_mlflow()
        setup_s3()

        if not setup_wandb():
            print("\n⚠️  WandB 설정에 실패했지만 계속 진행할 수 있습니다.")

        print("\n✅ MLOps 환경 설정 완료!")
        print("\n🚀 다음 단계:")
        print("   1. 파이프라인 테스트: python scripts/deploy_mlops.py test")
        print("   2. 애플리케이션 실행: python scripts/deploy_mlops.py run")

    elif args.command == "test":
        print("🧪 MLOps 파이프라인 테스트...")

        if not args.skip_checks and not check_environment():
            return 1

        success = run_pipeline_test()

        if success:
            print("\n✅ 테스트 완료! 애플리케이션을 실행할 준비가 되었습니다.")
            print("   실행 명령: python scripts/deploy_mlops.py run")
        else:
            print("\n❌ 테스트 실패. 설정을 확인하세요.")
            return 1

    elif args.command == "run":
        print("🚀 MLOps 애플리케이션 실행...")

        if not args.skip_checks and not check_environment():
            return 1

        setup_mlflow()  # MLflow 서버 시작

        success = run_mlops_app(args.port, args.dev)
        return 0 if success else 1

    elif args.command == "deploy":
        print("🚀 전체 MLOps 시스템 배포...")

        # 전체 설정 및 실행
        if not check_environment():
            return 1
        if not check_dependencies():
            return 1

        setup_mlflow()
        setup_s3()
        setup_wandb()

        print("\n🧪 파이프라인 테스트...")
        if not run_pipeline_test():
            print("❌ 파이프라인 테스트 실패. 애플리케이션 실행을 건너뜁니다.")
            return 1

        print("\n🚀 애플리케이션 실행...")
        success = run_mlops_app(args.port, args.dev)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)