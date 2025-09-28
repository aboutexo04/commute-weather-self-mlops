"""모델 훈련 스케줄링 시스템."""

from __future__ import annotations

import logging
import os
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
from zoneinfo import ZoneInfo

from ..config import KMAAPIConfig, ProjectPaths
from ..ml.enhanced_training import EnhancedCommutePredictor, EnhancedTrainingConfig
from ..ml.mlflow_integration import MLflowManager
from ..storage.s3_manager import S3StorageManager
from ..pipelines.ml.enhanced_flow import enhanced_mlops_pipeline, MLOpsConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingScheduleConfig:
    """훈련 스케줄 설정"""

    # 정기 훈련 주기
    daily_training_time: str = "02:00"  # 새벽 2시
    weekly_full_retrain_day: str = "sunday"  # 일요일
    weekly_full_retrain_time: str = "03:00"  # 새벽 3시

    # 성능 기반 훈련
    performance_check_interval_hours: int = 6  # 6시간마다 성능 체크
    performance_degradation_threshold: float = 0.15  # 15% 성능 저하시 재훈련

    # 데이터 기반 훈련
    min_new_data_samples: int = 50  # 새 데이터 50개 이상 시 재훈련
    data_quality_threshold: float = 0.8  # 데이터 품질 임계값

    # 안전 설정
    max_training_attempts_per_day: int = 3
    training_timeout_minutes: int = 60
    enable_emergency_fallback: bool = True

    # 알림 설정
    notify_on_training_start: bool = True
    notify_on_training_success: bool = True
    notify_on_training_failure: bool = True


class ModelTrainingScheduler:
    """모델 훈련 스케줄러"""

    def __init__(
        self,
        schedule_config: Optional[TrainingScheduleConfig] = None,
        mlops_config: Optional[MLOpsConfig] = None,
        project_root: Optional[Path] = None
    ):
        self.schedule_config = schedule_config or TrainingScheduleConfig()
        self.mlops_config = mlops_config or MLOpsConfig()
        self.project_root = project_root or Path.cwd()

        # 훈련 상태 추적
        self.training_attempts_today = 0
        self.last_training_date = None
        self.last_successful_training = None
        self.last_performance_check = None

        # 설정 로드
        self._load_configurations()

        logger.info("Model training scheduler initialized")

    def _load_configurations(self) -> None:
        """설정 정보 로드"""

        # 환경 변수에서 설정 로드
        self.kma_auth_key = os.getenv("KMA_AUTH_KEY")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.s3_bucket = os.getenv("COMMUTE_S3_BUCKET")
        self.wandb_api_key = os.getenv("WANDB_API_KEY")

        if not self.kma_auth_key:
            raise ValueError("KMA_AUTH_KEY environment variable is required")

        logger.info("Configurations loaded successfully")

    def setup_training_schedule(self) -> None:
        """훈련 스케줄 설정"""

        logger.info("Setting up training schedules")

        # 1. 일일 정기 훈련 (매일 새벽 2시)
        schedule.every().day.at(self.schedule_config.daily_training_time).do(
            self._scheduled_daily_training
        )

        # 2. 주간 전체 재훈련 (일요일 새벽 3시)
        getattr(schedule.every(), self.schedule_config.weekly_full_retrain_day).at(
            self.schedule_config.weekly_full_retrain_time
        ).do(self._scheduled_weekly_full_retrain)

        # 3. 성능 기반 체크 (6시간마다)
        schedule.every(self.schedule_config.performance_check_interval_hours).hours.do(
            self._scheduled_performance_check
        )

        # 4. 데이터 품질 체크 (2시간마다)
        schedule.every(2).hours.do(self._scheduled_data_quality_check)

        # 5. 일일 상태 리셋 (자정)
        schedule.every().day.at("00:00").do(self._reset_daily_counters)

        logger.info("Training schedules configured:")
        logger.info(f"  Daily training: {self.schedule_config.daily_training_time}")
        logger.info(f"  Weekly full retrain: {self.schedule_config.weekly_full_retrain_day} at {self.schedule_config.weekly_full_retrain_time}")
        logger.info(f"  Performance check: every {self.schedule_config.performance_check_interval_hours} hours")

    def _scheduled_daily_training(self) -> None:
        """일일 정기 훈련"""

        logger.info("🕐 Starting scheduled daily training")

        if not self._should_train_today():
            logger.info("Skipping daily training - limit reached")
            return

        try:
            self._notify_training_start("Daily Training")

            result = self._run_enhanced_mlops_pipeline("daily_training")

            if result.get("status") == "success":
                self.last_successful_training = datetime.now()
                self._notify_training_success("Daily Training", result)
                logger.info("✅ Daily training completed successfully")
            else:
                self._notify_training_failure("Daily Training", result)
                logger.error("❌ Daily training failed")

        except Exception as e:
            logger.error(f"Daily training error: {e}")
            self._notify_training_failure("Daily Training", {"error": str(e)})

        finally:
            self.training_attempts_today += 1

    def _scheduled_weekly_full_retrain(self) -> None:
        """주간 전체 재훈련"""

        logger.info("🗓️ Starting scheduled weekly full retrain")

        try:
            self._notify_training_start("Weekly Full Retrain")

            # 더 큰 데이터셋으로 전체 재훈련
            mlops_config = MLOpsConfig(
                min_observations_for_training=200,  # 더 큰 데이터셋
                retrain_interval_hours=168,  # 1주일
                enable_notifications=True
            )

            result = self._run_enhanced_mlops_pipeline("weekly_full_retrain", mlops_config)

            if result.get("status") == "success":
                self.last_successful_training = datetime.now()
                self._notify_training_success("Weekly Full Retrain", result)
                logger.info("✅ Weekly full retrain completed successfully")
            else:
                self._notify_training_failure("Weekly Full Retrain", result)
                logger.error("❌ Weekly full retrain failed")

        except Exception as e:
            logger.error(f"Weekly full retrain error: {e}")
            self._notify_training_failure("Weekly Full Retrain", {"error": str(e)})

    def _scheduled_performance_check(self) -> None:
        """성능 기반 체크"""

        logger.info("📊 Running scheduled performance check")

        try:
            self.last_performance_check = datetime.now()

            # MLflow에서 최근 성능 메트릭 조회
            mlflow_manager = MLflowManager(
                tracking_uri=self.mlflow_tracking_uri,
                experiment_name=self.mlops_config.experiment_name
            )

            # 현재 프로덕션 모델 성능 확인
            production_model = mlflow_manager.get_production_model("enhanced-commute-weather")

            if production_model:
                # 최근 성능 데이터와 비교
                best_run, model_uri = mlflow_manager.get_best_model_from_experiment()

                if best_run:
                    current_rmse = best_run.data.metrics.get("rmse", float('inf'))

                    # 성능 저하 임계값 확인
                    if current_rmse > self.schedule_config.performance_degradation_threshold:
                        logger.warning(f"Performance degradation detected: RMSE {current_rmse:.4f}")

                        if self._should_train_today():
                            logger.info("Triggering performance-based retraining")
                            self._notify_training_start("Performance-Based Retrain")

                            result = self._run_enhanced_mlops_pipeline("performance_based")

                            if result.get("status") == "success":
                                self._notify_training_success("Performance-Based Retrain", result)
                            else:
                                self._notify_training_failure("Performance-Based Retrain", result)

                            self.training_attempts_today += 1

        except Exception as e:
            logger.error(f"Performance check error: {e}")

    def _scheduled_data_quality_check(self) -> None:
        """데이터 품질 체크"""

        logger.info("🔍 Running scheduled data quality check")

        try:
            # S3에서 최근 데이터 품질 확인
            paths = ProjectPaths.from_root(self.project_root)
            storage = S3StorageManager(paths=paths, bucket=self.s3_bucket)

            # 최근 24시간 데이터 품질 메트릭 조회
            recent_files = storage.list_recent_quality_reports(hours_back=24)

            if recent_files:
                # 품질 점수 평균 계산
                quality_scores = []
                new_data_count = 0

                for file_path in recent_files[-10:]:  # 최근 10개 파일
                    try:
                        quality_report = storage.read_json(file_path)
                        if "feature_quality_score" in quality_report:
                            quality_scores.append(quality_report["feature_quality_score"])
                            new_data_count += quality_report.get("row_count", 0)
                    except Exception as e:
                        logger.warning(f"Failed to read quality report {file_path}: {e}")

                if quality_scores:
                    avg_quality = sum(quality_scores) / len(quality_scores)

                    logger.info(f"Data quality check: avg_quality={avg_quality:.3f}, new_samples={new_data_count}")

                    # 품질 임계값 및 새 데이터 량 체크
                    if (avg_quality >= self.schedule_config.data_quality_threshold and
                        new_data_count >= self.schedule_config.min_new_data_samples):

                        if self._should_train_today():
                            logger.info("Triggering data-based retraining")
                            self._notify_training_start("Data-Based Retrain")

                            result = self._run_enhanced_mlops_pipeline("data_based")

                            if result.get("status") == "success":
                                self._notify_training_success("Data-Based Retrain", result)
                            else:
                                self._notify_training_failure("Data-Based Retrain", result)

                            self.training_attempts_today += 1

        except Exception as e:
            logger.error(f"Data quality check error: {e}")

    def _run_enhanced_mlops_pipeline(
        self,
        trigger_type: str,
        custom_config: Optional[MLOpsConfig] = None
    ) -> Dict[str, Any]:
        """향상된 MLOps 파이프라인 실행"""

        logger.info(f"Running MLOps pipeline - trigger: {trigger_type}")

        config = custom_config or self.mlops_config

        # 파이프라인 실행
        result = enhanced_mlops_pipeline(
            project_root=self.project_root,
            kma_auth_key=self.kma_auth_key,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            s3_bucket=self.s3_bucket,
            config=config
        )

        # 트리거 유형 추가
        result["trigger_type"] = trigger_type
        result["timestamp"] = datetime.now().isoformat()

        return result

    def _should_train_today(self) -> bool:
        """오늘 훈련 가능 여부 확인"""

        today = datetime.now().date()

        # 날짜가 바뀌면 카운터 리셋
        if self.last_training_date != today:
            self.training_attempts_today = 0
            self.last_training_date = today

        return self.training_attempts_today < self.schedule_config.max_training_attempts_per_day

    def _reset_daily_counters(self) -> None:
        """일일 카운터 리셋"""
        self.training_attempts_today = 0
        self.last_training_date = datetime.now().date()
        logger.info("Daily training counters reset")

    def _notify_training_start(self, training_type: str) -> None:
        """훈련 시작 알림"""
        if not self.schedule_config.notify_on_training_start:
            return

        try:
            import wandb

            wandb.init(
                project="commute-weather-training-schedule",
                name=f"{training_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["scheduled", "training", training_type.lower()]
            )

            wandb.log({
                "event": "training_start",
                "training_type": training_type,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"🔔 Training start notification sent: {training_type}")

        except Exception as e:
            logger.warning(f"Failed to send training start notification: {e}")

    def _notify_training_success(self, training_type: str, result: Dict[str, Any]) -> None:
        """훈련 성공 알림"""
        if not self.schedule_config.notify_on_training_success:
            return

        try:
            import wandb

            wandb.log({
                "event": "training_success",
                "training_type": training_type,
                "models_trained": result.get("training_result", {}).get("models_trained", 0),
                "best_model": result.get("training_result", {}).get("best_model", "unknown"),
                "timestamp": datetime.now().isoformat()
            })

            wandb.finish()

            logger.info(f"✅ Training success notification sent: {training_type}")

        except Exception as e:
            logger.warning(f"Failed to send training success notification: {e}")

    def _notify_training_failure(self, training_type: str, result: Dict[str, Any]) -> None:
        """훈련 실패 알림"""
        if not self.schedule_config.notify_on_training_failure:
            return

        try:
            import wandb

            wandb.init(
                project="commute-weather-training-schedule",
                name=f"failure_{training_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["scheduled", "training", "failure"]
            )

            wandb.log({
                "event": "training_failure",
                "training_type": training_type,
                "error": result.get("error", "unknown"),
                "timestamp": datetime.now().isoformat()
            })

            wandb.finish()

            logger.error(f"❌ Training failure notification sent: {training_type}")

        except Exception as e:
            logger.warning(f"Failed to send training failure notification: {e}")

    def run_scheduler(self) -> None:
        """스케줄러 실행"""

        logger.info("🚀 Starting model training scheduler")

        self.setup_training_schedule()

        logger.info("Scheduler is running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise

    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""

        return {
            "training_attempts_today": self.training_attempts_today,
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
            "last_successful_training": self.last_successful_training.isoformat() if self.last_successful_training else None,
            "last_performance_check": self.last_performance_check.isoformat() if self.last_performance_check else None,
            "next_scheduled_jobs": [str(job) for job in schedule.jobs],
            "max_daily_attempts": self.schedule_config.max_training_attempts_per_day,
            "can_train_today": self._should_train_today()
        }


def create_training_scheduler(
    schedule_config: Optional[TrainingScheduleConfig] = None,
    mlops_config: Optional[MLOpsConfig] = None,
    project_root: Optional[Path] = None
) -> ModelTrainingScheduler:
    """훈련 스케줄러 생성"""

    return ModelTrainingScheduler(
        schedule_config=schedule_config,
        mlops_config=mlops_config,
        project_root=project_root
    )