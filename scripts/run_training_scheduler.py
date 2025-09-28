#!/usr/bin/env python3
"""모델 훈련 스케줄러 실행 스크립트."""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from commute_weather.scheduling import (
    ModelTrainingScheduler,
    TrainingScheduleConfig,
    create_training_scheduler
)
from commute_weather.pipelines.ml.enhanced_flow import MLOpsConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/training_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """명령행 인수 파싱"""

    parser = argparse.ArgumentParser(description="Model Training Scheduler")

    parser.add_argument(
        "--daily-time",
        default="02:00",
        help="Daily training time (HH:MM format, default: 02:00)"
    )

    parser.add_argument(
        "--weekly-day",
        default="sunday",
        choices=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        help="Weekly full retrain day (default: sunday)"
    )

    parser.add_argument(
        "--weekly-time",
        default="03:00",
        help="Weekly full retrain time (HH:MM format, default: 03:00)"
    )

    parser.add_argument(
        "--performance-check-hours",
        type=int,
        default=6,
        help="Performance check interval in hours (default: 6)"
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum training attempts per day (default: 3)"
    )

    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=0.15,
        help="Performance degradation threshold (default: 0.15)"
    )

    parser.add_argument(
        "--min-data-samples",
        type=int,
        default=50,
        help="Minimum new data samples for retraining (default: 50)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show scheduler status and exit"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show schedule configuration without running"
    )

    return parser.parse_args()


def create_schedule_config(args) -> TrainingScheduleConfig:
    """명령행 인수로부터 스케줄 설정 생성"""

    return TrainingScheduleConfig(
        daily_training_time=args.daily_time,
        weekly_full_retrain_day=args.weekly_day,
        weekly_full_retrain_time=args.weekly_time,
        performance_check_interval_hours=args.performance_check_hours,
        performance_degradation_threshold=args.performance_threshold,
        min_new_data_samples=args.min_data_samples,
        max_training_attempts_per_day=args.max_attempts,
        notify_on_training_start=True,
        notify_on_training_success=True,
        notify_on_training_failure=True
    )


def create_mlops_config() -> MLOpsConfig:
    """MLOps 설정 생성"""

    return MLOpsConfig(
        lookback_hours=3,
        min_observations_for_training=100,
        data_quality_threshold=0.7,
        retrain_interval_hours=24,
        performance_threshold=0.1,
        experiment_name="scheduled-commute-mlops",
        model_stage="Production",
        enable_notifications=True,
        performance_degradation_threshold=0.15
    )


def show_schedule_configuration(schedule_config: TrainingScheduleConfig, mlops_config: MLOpsConfig):
    """스케줄 설정 표시"""

    print("\n" + "="*60)
    print("🕐 MODEL TRAINING SCHEDULE CONFIGURATION")
    print("="*60)

    print(f"\n📅 Regular Training Schedule:")
    print(f"   Daily Training: Every day at {schedule_config.daily_training_time}")
    print(f"   Weekly Full Retrain: Every {schedule_config.weekly_full_retrain_day} at {schedule_config.weekly_full_retrain_time}")

    print(f"\n🔍 Monitoring Schedule:")
    print(f"   Performance Check: Every {schedule_config.performance_check_interval_hours} hours")
    print(f"   Data Quality Check: Every 2 hours")

    print(f"\n⚙️ Trigger Conditions:")
    print(f"   Performance Degradation: >{schedule_config.performance_degradation_threshold*100:.1f}%")
    print(f"   New Data Samples: >{schedule_config.min_new_data_samples} samples")
    print(f"   Data Quality Threshold: >{schedule_config.data_quality_threshold*100:.1f}%")

    print(f"\n🛡️ Safety Limits:")
    print(f"   Max Training Attempts/Day: {schedule_config.max_training_attempts_per_day}")
    print(f"   Training Timeout: {schedule_config.training_timeout_minutes} minutes")

    print(f"\n📊 MLOps Configuration:")
    print(f"   Min Training Samples: {mlops_config.min_observations_for_training}")
    print(f"   Experiment Name: {mlops_config.experiment_name}")
    print(f"   Model Stage: {mlops_config.model_stage}")

    print(f"\n🔔 Notification Settings:")
    print(f"   Training Start: {'✅' if schedule_config.notify_on_training_start else '❌'}")
    print(f"   Training Success: {'✅' if schedule_config.notify_on_training_success else '❌'}")
    print(f"   Training Failure: {'✅' if schedule_config.notify_on_training_failure else '❌'}")

    print("="*60)


def main():
    """메인 실행 함수"""

    args = parse_arguments()

    # 프로젝트 루트 경로
    project_root = Path(__file__).resolve().parents[1]

    # 설정 생성
    schedule_config = create_schedule_config(args)
    mlops_config = create_mlops_config()

    # 설정 표시
    show_schedule_configuration(schedule_config, mlops_config)

    # Dry run 모드
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Configuration shown above")
        print("💡 Use without --dry-run to start the scheduler")
        return

    try:
        # 스케줄러 생성
        scheduler = create_training_scheduler(
            schedule_config=schedule_config,
            mlops_config=mlops_config,
            project_root=project_root
        )

        # 상태 확인 모드
        if args.status:
            status = scheduler.get_scheduler_status()

            print(f"\n📊 SCHEDULER STATUS")
            print(f"   Training Attempts Today: {status['training_attempts_today']}")
            print(f"   Last Training Date: {status['last_training_date'] or 'Never'}")
            print(f"   Last Successful Training: {status['last_successful_training'] or 'Never'}")
            print(f"   Last Performance Check: {status['last_performance_check'] or 'Never'}")
            print(f"   Can Train Today: {'✅' if status['can_train_today'] else '❌'}")

            print(f"\n📋 Scheduled Jobs:")
            for i, job in enumerate(status['next_scheduled_jobs'], 1):
                print(f"   {i}. {job}")

            return

        # 환경변수 확인
        required_env_vars = ["KMA_AUTH_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
            print("Please set them before running the scheduler.")
            sys.exit(1)

        optional_env_vars = {
            "MLFLOW_TRACKING_URI": "file:./mlruns",
            "COMMUTE_S3_BUCKET": "Not configured",
            "WANDB_API_KEY": "Not configured"
        }

        print(f"\n🔧 Environment Configuration:")
        for var, default in optional_env_vars.items():
            value = os.getenv(var, default)
            print(f"   {var}: {value}")

        print(f"\n🚀 Starting Model Training Scheduler...")
        print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Project Root: {project_root}")

        # 스케줄러 실행
        scheduler.run_scheduler()

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\n🛑 Scheduler stopped by user")

    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        print(f"\n❌ Scheduler failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()