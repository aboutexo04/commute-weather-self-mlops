"""향상된 MLOps 파이프라인 - KMA 데이터 기반 완전 자동화."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from prefect import flow, task, get_run_logger
from zoneinfo import ZoneInfo

from ...config import KMAAPIConfig, ProjectPaths
from ...data_sources.kma_api import fetch_recent_weather_kma
from ...data_sources.weather_api import WeatherObservation
from ...features.feature_engineering import WeatherFeatureEngineer
from ...ml.enhanced_training import EnhancedCommutePredictor, EnhancedTrainingConfig
from ...ml.mlflow_integration import MLflowManager
from ...storage.s3_manager import S3StorageManager
from ...pipelines.baseline import compute_commute_comfort_score

logger = logging.getLogger(__name__)


@dataclass
class MLOpsConfig:
    """MLOps 파이프라인 설정"""

    # 데이터 수집
    lookback_hours: int = 3
    min_observations_for_training: int = 100
    data_quality_threshold: float = 0.7

    # 모델 훈련
    retrain_interval_hours: int = 24  # 24시간마다 재훈련
    performance_threshold: float = 0.1  # RMSE 임계값

    # 실험 관리
    experiment_name: str = "enhanced-commute-mlops"
    model_stage: str = "Production"

    # 알림 설정
    enable_notifications: bool = True
    performance_degradation_threshold: float = 0.15


@task
def collect_recent_kma_data(
    kma_config: KMAAPIConfig,
    hours_back: int = 72  # 3일치 데이터
) -> List[WeatherObservation]:
    """최근 KMA 데이터 수집"""

    logger = get_run_logger()
    logger.info(f"Collecting KMA data for last {hours_back} hours")

    try:
        observations = fetch_recent_weather_kma(
            kma_config,
            lookback_hours=hours_back
        )

        logger.info(f"Collected {len(observations)} weather observations")
        return observations

    except Exception as e:
        logger.error(f"Failed to collect KMA data: {e}")
        raise


@task
def load_real_time_training_data(
    storage: S3StorageManager,
    days_back: int = 30
) -> Tuple[List[List[WeatherObservation]], List[float], Dict[str, Any]]:
    """실시간 수집된 데이터로부터 훈련 데이터 로드"""

    logger = get_run_logger()
    logger.info(f"Loading real-time training data from last {days_back} days")

    try:
        from ...data.training_data_manager import RealTimeTrainingDataManager

        # 훈련 데이터 매니저 생성
        data_manager = RealTimeTrainingDataManager(
            storage_manager=storage,
            min_quality_score=0.6,
            min_observations_per_set=3
        )

        # 실시간 수집 데이터로부터 훈련 데이터 구성
        observation_sets, comfort_scores, stats = data_manager.collect_training_data(
            days_back=days_back,
            station_id="108"
        )

        # 데이터셋 요약 생성
        dataset_summary = data_manager.create_training_dataset_summary(
            observation_sets, comfort_scores, stats
        )

        logger.info(f"Loaded {len(observation_sets)} real-time training samples")
        logger.info(f"Average quality score: {stats.average_quality_score:.3f}")
        logger.info(f"Data completeness: {stats.data_completeness:.3f}")

        return observation_sets, comfort_scores, dataset_summary

    except Exception as e:
        logger.error(f"Failed to load real-time training data: {e}")
        return [], [], {"error": str(e)}


@task
def validate_training_data_quality(
    observation_sets: List[List[WeatherObservation]],
    comfort_scores: List[float],
    dataset_summary: Dict[str, Any],
    min_samples: int = 100
) -> Dict[str, Any]:
    """훈련 데이터 품질 검증"""

    logger = get_run_logger()
    logger.info("Validating training data quality")

    validation_result = {
        "is_valid": True,
        "issues": [],
        "recommendations": [],
        "stats": dataset_summary
    }

    # 1. 최소 샘플 수 확인
    if len(observation_sets) < min_samples:
        validation_result["is_valid"] = False
        validation_result["issues"].append(f"Insufficient samples: {len(observation_sets)} < {min_samples}")
        validation_result["recommendations"].append("Collect more historical data or reduce min_samples")

    # 2. 데이터 품질 확인
    avg_quality = dataset_summary.get("dataset_info", {}).get("data_quality", {}).get("average_quality_score", 0)
    if avg_quality < 0.7:
        validation_result["issues"].append(f"Low average data quality: {avg_quality:.3f}")
        validation_result["recommendations"].append("Improve data collection or lower quality threshold")

    # 3. 타겟 분포 확인
    target_stats = dataset_summary.get("target_distribution", {})
    comfort_std = target_stats.get("comfort_score_std", 0)
    if comfort_std < 0.1:
        validation_result["issues"].append(f"Low target variance: {comfort_std:.3f}")
        validation_result["recommendations"].append("Ensure diverse weather conditions in training data")

    # 4. 데이터 완성도 확인
    completeness = dataset_summary.get("dataset_info", {}).get("data_quality", {}).get("data_completeness", 0)
    if completeness < 0.5:
        validation_result["issues"].append(f"Low data completeness: {completeness:.3f}")
        validation_result["recommendations"].append("Check data collection pipeline for gaps")

    logger.info(f"Training data validation: {'✅ PASSED' if validation_result['is_valid'] else '❌ FAILED'}")
    if validation_result["issues"]:
        for issue in validation_result["issues"]:
            logger.warning(f"Issue: {issue}")

    return validation_result


@task
def train_enhanced_models(
    observations_sets: List[List[WeatherObservation]],
    comfort_scores: List[float],
    dataset_summary: Dict[str, Any],
    mlflow_manager: MLflowManager,
    s3_manager: S3StorageManager,
    config: MLOpsConfig
) -> Dict[str, Any]:
    """향상된 모델 훈련"""

    logger = get_run_logger()
    logger.info("Starting enhanced model training")

    if len(observations_sets) < config.min_observations_for_training:
        logger.warning(f"Insufficient training data: {len(observations_sets)} < {config.min_observations_for_training}")
        return {"status": "skipped", "reason": "insufficient_data"}

    try:
        # 훈련 설정
        training_config = EnhancedTrainingConfig(
            min_training_samples=config.min_observations_for_training,
            enable_feature_selection=True,
            enable_hyperparameter_tuning=True,
            use_time_series_split=True
        )

        # 예측기 생성
        predictor = EnhancedCommutePredictor(
            config=training_config,
            mlflow_manager=mlflow_manager,
            s3_manager=s3_manager
        )

        # 타임스탬프 생성
        timestamps = [obs[-1].timestamp for obs in observations_sets]

        # 훈련 및 추적
        run_ids = predictor.train_and_track(
            observations_list=observations_sets,
            comfort_scores=comfort_scores,
            timestamps=timestamps,
            experiment_name=config.experiment_name
        )

        # 최고 성능 모델 정보
        best_model = predictor.best_model_name
        best_rmse = None

        if best_model and hasattr(predictor, 'models'):
            # 성능 정보 추출 (실제 구현에서는 results에서 가져옴)
            logger.info(f"Best model: {best_model}")

        result = {
            "status": "success",
            "models_trained": len(run_ids),
            "best_model": best_model,
            "run_ids": run_ids,
            "training_samples": len(observations_sets),
            "feature_count": len(predictor.selected_features) if predictor.selected_features else 0,
            "data_source": "real_time_kma_collection",
            "dataset_summary": dataset_summary
        }

        logger.info(f"Training completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {"status": "failed", "error": str(e)}


@task
def evaluate_model_performance(
    mlflow_manager: MLflowManager,
    current_performance: Dict[str, float],
    config: MLOpsConfig
) -> Dict[str, Any]:
    """모델 성능 평가 및 품질 검사"""

    logger = get_run_logger()
    logger.info("Evaluating model performance")

    try:
        # 현재 프로덕션 모델 성능 가져오기
        production_model = mlflow_manager.get_production_model("enhanced-commute-weather")

        evaluation_result = {
            "current_rmse": current_performance.get("rmse", float('inf')),
            "performance_degraded": False,
            "needs_retraining": False,
            "production_model_exists": production_model is not None
        }

        # 프로덕션 모델과 비교
        if production_model:
            # 최근 성능 메트릭 가져오기 (MLflow에서)
            best_run, model_uri = mlflow_manager.get_best_model_from_experiment()

            if best_run:
                production_rmse = best_run.data.metrics.get("rmse", float('inf'))
                current_rmse = current_performance.get("rmse", float('inf'))

                # 성능 저하 확인
                performance_degradation = (current_rmse - production_rmse) / production_rmse

                if performance_degradation > config.performance_degradation_threshold:
                    evaluation_result["performance_degraded"] = True
                    evaluation_result["needs_retraining"] = True

                evaluation_result["production_rmse"] = production_rmse
                evaluation_result["performance_change"] = performance_degradation

        # 성능 임계값 확인
        if current_performance.get("rmse", float('inf')) > config.performance_threshold:
            evaluation_result["needs_retraining"] = True

        logger.info(f"Performance evaluation: {evaluation_result}")
        return evaluation_result

    except Exception as e:
        logger.error(f"Performance evaluation failed: {e}")
        return {"status": "failed", "error": str(e)}


@task
def promote_best_model(
    mlflow_manager: MLflowManager,
    training_result: Dict[str, Any],
    evaluation_result: Dict[str, Any],
    config: MLOpsConfig
) -> Dict[str, Any]:
    """최고 성능 모델을 프로덕션으로 승격"""

    logger = get_run_logger()

    if training_result.get("status") != "success":
        logger.info("Skipping model promotion - training was not successful")
        return {"status": "skipped", "reason": "training_failed"}

    if not evaluation_result.get("needs_retraining", False):
        logger.info("Skipping model promotion - current model performance is acceptable")
        return {"status": "skipped", "reason": "performance_acceptable"}

    try:
        logger.info("Promoting best model to production")

        # 최고 성능 모델 찾기
        best_run, model_uri = mlflow_manager.get_best_model_from_experiment()

        if not best_run:
            logger.warning("No best model found")
            return {"status": "failed", "reason": "no_best_model"}

        # 모델 등록 및 승격
        model_name = "enhanced-commute-weather"

        # 모델 버전 등록
        registered_model = mlflow_manager.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=best_run.info.run_id
        )

        # 프로덕션 스테이지로 승격
        mlflow_manager.promote_model_to_stage(
            model_name=model_name,
            version=registered_model.version,
            stage=config.model_stage
        )

        result = {
            "status": "success",
            "model_name": model_name,
            "version": registered_model.version,
            "stage": config.model_stage,
            "run_id": best_run.info.run_id,
            "rmse": best_run.data.metrics.get("rmse")
        }

        logger.info(f"Model promotion completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        return {"status": "failed", "error": str(e)}


@task
def send_mlops_notification(
    training_result: Dict[str, Any],
    evaluation_result: Dict[str, Any],
    promotion_result: Dict[str, Any],
    config: MLOpsConfig
) -> None:
    """MLOps 파이프라인 결과 알림"""

    logger = get_run_logger()

    if not config.enable_notifications:
        return

    try:
        # WandB 알림 (설정된 경우)
        import wandb

        wandb.init(
            project="commute-weather-mlops",
            name=f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["mlops", "automated"]
        )

        # 메트릭 로깅
        wandb.log({
            "training_status": 1 if training_result.get("status") == "success" else 0,
            "models_trained": training_result.get("models_trained", 0),
            "training_samples": training_result.get("training_samples", 0),
            "performance_degraded": 1 if evaluation_result.get("performance_degraded") else 0,
            "model_promoted": 1 if promotion_result.get("status") == "success" else 0
        })

        # 알림 메시지 생성
        if training_result.get("status") == "success":
            message = f"✅ MLOps Pipeline Success: {training_result.get('models_trained', 0)} models trained"

            if promotion_result.get("status") == "success":
                message += f", Model v{promotion_result.get('version')} promoted to production"

        else:
            message = f"❌ MLOps Pipeline Failed: {training_result.get('error', 'Unknown error')}"

        wandb.log({"notification": message})
        wandb.finish()

        logger.info(f"Notification sent: {message}")

    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


@flow(name="enhanced-commute-mlops-pipeline")
def enhanced_mlops_pipeline(
    project_root: Optional[Path] = None,
    kma_auth_key: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    config: Optional[MLOpsConfig] = None
) -> Dict[str, Any]:
    """향상된 MLOps 파이프라인 - 완전 자동화"""

    logger = get_run_logger()
    logger.info("Starting Enhanced MLOps Pipeline")

    # 기본 설정
    if config is None:
        config = MLOpsConfig()

    # 프로젝트 경로 설정
    if project_root is None:
        project_root = Path(__file__).resolve().parents[4]

    paths = ProjectPaths.from_root(project_root)

    # KMA API 설정
    auth_key = kma_auth_key or os.getenv("KMA_AUTH_KEY")
    if not auth_key:
        raise ValueError("KMA API key required")

    kma_config = KMAAPIConfig(
        base_url="https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
        auth_key=auth_key,
        station_id="108"  # 서울
    )

    # 스토리지 설정
    s3_bucket = s3_bucket or os.getenv("COMMUTE_S3_BUCKET")
    storage = S3StorageManager(paths=paths, bucket=s3_bucket)

    # MLflow 설정
    tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_manager = MLflowManager(
        tracking_uri=tracking_uri,
        experiment_name=config.experiment_name
    )

    # 파이프라인 실행
    try:
        # 1. 최근 데이터 수집
        recent_observations = collect_recent_kma_data(kma_config, hours_back=72)

        # 2. 실시간 수집 데이터 로드 및 라벨 생성
        historical_observations, comfort_scores, dataset_summary = load_real_time_training_data(storage, days_back=30)

        # 3. 훈련 데이터 품질 검증
        validation_result = validate_training_data_quality(
            historical_observations, comfort_scores, dataset_summary, config.min_observations_for_training
        )

        # 4. 모델 훈련 (검증 통과 시에만)
        if not validation_result["is_valid"]:
            logger.warning("Training data validation failed - skipping training")
            training_result = {
                "status": "skipped",
                "reason": "data_validation_failed",
                "validation_issues": validation_result["issues"]
            }
        else:
            training_result = train_enhanced_models(
                historical_observations,
                comfort_scores,
                dataset_summary,
                mlflow_manager,
                storage,
                config
            )

        # 5. 성능 평가
        current_performance = {"rmse": 0.1}  # 실제로는 training_result에서 추출
        evaluation_result = evaluate_model_performance(
            mlflow_manager,
            current_performance,
            config
        )

        # 6. 모델 승격
        promotion_result = promote_best_model(
            mlflow_manager,
            training_result,
            evaluation_result,
            config
        )

        # 7. 알림 발송
        send_mlops_notification(
            training_result,
            evaluation_result,
            promotion_result,
            config
        )

        # 결과 정리
        pipeline_result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "recent_observations": len(recent_observations),
            "historical_sets": len(historical_observations),
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "promotion_result": promotion_result,
            "pipeline_config": config.__dict__
        }

        logger.info(f"Enhanced MLOps Pipeline completed successfully: {pipeline_result}")
        return pipeline_result

    except Exception as e:
        logger.error(f"Enhanced MLOps Pipeline failed: {e}")

        # 실패 알림
        if config.enable_notifications:
            try:
                import wandb
                wandb.init(project="commute-weather-mlops", name="pipeline_failure")
                wandb.log({"error": str(e), "status": "failed"})
                wandb.finish()
            except:
                pass

        raise


if __name__ == "__main__":
    # 테스트 실행
    result = enhanced_mlops_pipeline()
    print(f"Pipeline result: {result}")