#!/usr/bin/env python3
"""실시간 MLOps 파이프라인 실행 스크립트."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from commute_weather.pipelines.realtime_mlops_pipeline import (
    realtime_mlops_flow,
    hourly_asos_collection_flow
)
from commute_weather.pipelines.evaluation_pipeline import model_evaluation_flow

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_realtime_pipeline(args):
    """실시간 MLOps 파이프라인 실행"""
    print("🚀 Starting realtime MLOps pipeline...")

    try:
        result = realtime_mlops_flow(
            auth_key=args.auth_key,
            station_id=args.station_id,
            use_s3=args.use_s3,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            mlflow_tracking_uri=args.mlflow_uri,
            feature_version=args.feature_version,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )

        if result["status"] == "success":
            print("✅ Realtime MLOps pipeline completed successfully!")
            print(f"📊 Prediction: {result.get('prediction_value', 'N/A'):.2f}")
            print(f"🎯 Confidence: {result.get('prediction_confidence', 0):.3f}")
            print(f"⏱️  Execution time: {result.get('execution_time_seconds', 0):.2f}s")
            print(f"🗓️  Data timestamp: {result.get('data_timestamp', 'N/A')}")

            if args.verbose:
                print("\n📈 Detailed Results:")
                print(f"  - Data quality: {result.get('data_quality', 0):.1f}")
                print(f"  - Feature quality: {result.get('feature_quality', 0):.1f}")
                print(f"  - Model used: {result.get('model_used', 'N/A')}")

                s3_keys = result.get("s3_keys", {})
                if s3_keys.get("features"):
                    print(f"  - Features S3: {s3_keys['features']}")
                if s3_keys.get("prediction"):
                    print(f"  - Prediction S3: {s3_keys['prediction']}")

        else:
            print("❌ Realtime MLOps pipeline failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1

    return 0


def run_hourly_collection(args):
    """매시간 ASOS 데이터 수집 실행"""
    print(f"📡 Starting hourly ASOS data collection ({args.hours_back} hours)...")

    try:
        result = hourly_asos_collection_flow(
            auth_key=args.auth_key,
            station_id=args.station_id,
            hours_back=args.hours_back,
            use_s3=args.use_s3,
            s3_bucket=args.s3_bucket
        )

        if result["status"] == "success":
            print("✅ Hourly ASOS collection completed successfully!")
            print(f"📊 Collected: {result.get('collected_count', 0)} observations")
            print(f"💾 Stored files: {len(result.get('stored_files', []))}")

            if args.verbose:
                print("\n📁 Stored Files:")
                for file_info in result.get("stored_files", []):
                    if file_info["type"] == "s3":
                        print(f"  - S3: {file_info['key']}")
                    else:
                        print(f"  - Local: {file_info['path']}")

        else:
            print("❌ Hourly ASOS collection failed!")
            return 1

    except Exception as e:
        print(f"❌ Collection failed: {e}")
        logger.error(f"Collection failed: {e}", exc_info=True)
        return 1

    return 0


def run_model_evaluation(args):
    """모델 평가 실행"""
    print(f"🔍 Starting model evaluation (last {args.hours_back} hours)...")

    try:
        result = model_evaluation_flow(
            auth_key=args.auth_key,
            station_id=args.station_id,
            hours_back=args.hours_back,
            use_s3=args.use_s3,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            mlflow_tracking_uri=args.mlflow_uri,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )

        if result["status"] == "success":
            print("✅ Model evaluation completed successfully!")
            print(f"📊 Model: {result.get('model_name', 'N/A')} v{result.get('model_version', 'N/A')}")
            print(f"🎯 R² Score: {result['performance_metrics'].get('r2_score', 0):.3f}")
            print(f"📈 MAE: {result['performance_metrics'].get('mae', 0):.3f}")
            print(f"⚡ Valid: {result['validation_results'].get('is_valid', False)}")
            print(f"🔄 Retraining Required: {result.get('retraining_required', False)}")
            print(f"⏱️  Execution time: {result.get('execution_time_seconds', 0):.2f}s")

            if args.verbose and result.get('recommendations'):
                print("\n📋 Recommendations:")
                for rec in result['recommendations'][:3]:  # 상위 3개만 표시
                    print(f"  - {rec}")

        else:
            print("❌ Model evaluation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"❌ Evaluation execution failed: {e}")
        logger.error(f"Evaluation execution failed: {e}", exc_info=True)
        return 1

    return 0


def run_continuous_mode(args):
    """연속 실행 모드"""
    print(f"🔄 Starting continuous mode (every {args.interval} minutes)...")

    import time
    import schedule

    def job():
        """스케줄된 작업"""
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Running scheduled pipeline...")

        if args.mode == "realtime":
            return run_realtime_pipeline(args)
        elif args.mode == "hourly":
            return run_hourly_collection(args)
        elif args.mode == "evaluation":
            return run_model_evaluation(args)

    # 스케줄 설정
    if args.interval == 60:  # 1시간마다
        schedule.every().hour.do(job)
        print("📅 Scheduled to run every hour")
    else:
        schedule.every(args.interval).minutes.do(job)
        print(f"📅 Scheduled to run every {args.interval} minutes")

    # 즉시 한 번 실행
    if args.run_immediately:
        print("🚀 Running immediately...")
        job()

    # 연속 실행
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 스케줄 확인

    except KeyboardInterrupt:
        print("\n🛑 Continuous mode stopped by user")
        return 0


def main():
    parser = argparse.ArgumentParser(description="실시간 MLOps 파이프라인 실행")

    # 기본 설정
    parser.add_argument("mode", choices=["realtime", "hourly", "continuous", "evaluation"], help="실행 모드")
    parser.add_argument("--auth-key", help="KMA API 인증 키")
    parser.add_argument("--station-id", default="108", help="기상 관측소 ID (기본: 108-서울)")

    # S3 설정
    parser.add_argument("--use-s3", action="store_true", default=True, help="S3 사용 여부")
    parser.add_argument("--s3-bucket", help="S3 버킷 이름")
    parser.add_argument("--s3-prefix", default="", help="S3 키 접두사")

    # MLflow 설정
    parser.add_argument("--mlflow-uri", help="MLflow 추적 URI")
    parser.add_argument("--feature-version", default="v2", help="피처 버전")

    # WandB 설정
    parser.add_argument("--use-wandb", action="store_true", default=True, help="WandB 사용 여부")
    parser.add_argument("--wandb-project", default="commute-weather-mlops", help="WandB 프로젝트 이름")
    parser.add_argument("--wandb-entity", help="WandB 엔티티 (팀 또는 사용자명)")

    # 시간 설정
    parser.add_argument("--hours-back", type=int, default=1, help="수집할 과거 시간 수 (hourly 모드용)")

    # 연속 실행 설정
    parser.add_argument("--interval", type=int, default=60, help="연속 모드 실행 간격 (분)")
    parser.add_argument("--run-immediately", action="store_true", help="연속 모드에서 즉시 실행")

    # 출력 설정
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")

    args = parser.parse_args()

    # 환경변수에서 설정 값 확인
    if not args.auth_key:
        args.auth_key = os.getenv("KMA_AUTH_KEY")
        if not args.auth_key:
            print("❌ KMA API 인증 키가 필요합니다. --auth-key 옵션이나 KMA_AUTH_KEY 환경변수를 설정하세요.")
            return 1

    if not args.s3_bucket:
        args.s3_bucket = os.getenv("COMMUTE_S3_BUCKET")
        if args.use_s3 and not args.s3_bucket:
            print("⚠️  S3 버킷이 지정되지 않았습니다. 로컬 저장소만 사용합니다.")
            args.use_s3 = False

    # 모드별 실행
    if args.mode == "continuous":
        return run_continuous_mode(args)
    elif args.mode == "realtime":
        return run_realtime_pipeline(args)
    elif args.mode == "hourly":
        return run_hourly_collection(args)
    elif args.mode == "evaluation":
        return run_model_evaluation(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)