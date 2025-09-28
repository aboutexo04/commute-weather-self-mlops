"""강화된 모델 훈련 파이프라인 with MLflow 및 S3 통합."""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from prefect import flow, task, get_run_logger
from zoneinfo import ZoneInfo

from ..config import ProjectPaths
from ..storage import StorageManager
from ..storage.s3_manager import S3StorageManager
from ..ml import ModelFactory, compare_models, get_best_model, TrainingResult
from ..ml.mlflow_integration import MLflowManager
from ..data_quality import DataQualityValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifact:
    """훈련 결과 아티팩트"""
    best_model_type: str
    best_model_s3_key: Optional[str]
    mlflow_run_id: str
    model_comparison_results: Dict[str, str]  # model_type -> run_id
    training_data_quality_score: float
    training_samples: int
    validation_samples: int
    best_model_metrics: Dict[str, float]
    timestamp: dt.datetime


@task
def collect_training_data(
    storage: StorageManager,
    feature_version: str = "v1",
    min_samples: int = 24,
    lookback_days: int = 30
) -> Dict[str, Any]:
    """훈련 데이터 수집 및 품질 검증"""
    logger = get_run_logger()

    # 피처 파일 목록 가져오기
    feature_files = storage.list_feature_files(feature_version)

    if not feature_files:
        raise ValueError(f"No feature files found for version {feature_version}")

    logger.info(f"Found {len(feature_files)} feature files")

    # 최근 데이터만 선택 (lookback_days 내)
    cutoff_date = dt.datetime.now(ZoneInfo("Asia/Seoul")) - dt.timedelta(days=lookback_days)

    # 파일에서 데이터 로드
    dataframes = []
    for file_path in feature_files:
        try:
            df = storage.read_parquet(file_path)
            if not df.empty:
                dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    if not dataframes:
        raise ValueError("No valid feature data found")

    # 전체 데이터 결합
    combined_df = pd.concat(dataframes, ignore_index=True)

    # 시간 필터링
    if "event_time" in combined_df.columns:
        combined_df["event_time"] = pd.to_datetime(combined_df["event_time"])
        mask = combined_df["event_time"] >= cutoff_date
        combined_df = combined_df[mask]

    # 중복 제거 (같은 시간의 데이터)
    if "event_time" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["event_time"], keep="last")

    # 데이터 정렬
    if "event_time" in combined_df.columns:
        combined_df = combined_df.sort_values("event_time").reset_index(drop=True)

    logger.info(f"Collected {len(combined_df)} training samples")

    # 최소 샘플 수 확인
    if len(combined_df) < min_samples:
        raise ValueError(
            f"Insufficient training data: {len(combined_df)} < {min_samples} samples"
        )

    # 데이터 품질 검증
    validator = DataQualityValidator()

    # 기본 검증을 위한 컬럼 매핑
    weather_df = combined_df.copy()
    if "temp_median_3h" in weather_df.columns:
        weather_df["temperature_c"] = weather_df["temp_median_3h"]
    if "wind_peak_3h" in weather_df.columns:
        weather_df["wind_speed_ms"] = weather_df["wind_peak_3h"]
    if "precip_total_3h" in weather_df.columns:
        weather_df["precipitation_mm"] = weather_df["precip_total_3h"]
    if "humidity_avg_3h" in weather_df.columns:
        weather_df["relative_humidity"] = weather_df["humidity_avg_3h"]

    # 가짜 타임스탬프 추가 (검증용)
    if "timestamp" not in weather_df.columns and "event_time" in weather_df.columns:
        weather_df["timestamp"] = weather_df["event_time"]

    validation_results = validator.validate_weather_data(weather_df)
    quality_summary = validator.get_summary()

    logger.info(
        f"Training data quality: {quality_summary['overall_quality']} "
        f"({quality_summary['passed_checks']}/{quality_summary['total_checks']} checks passed)"
    )

    # 품질 점수 계산
    total_checks = quality_summary['total_checks']
    passed_checks = quality_summary['passed_checks']
    quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100.0

    return {
        "dataframe": combined_df,
        "quality_score": quality_score,
        "quality_summary": quality_summary,
        "validation_results": validation_results,
        "total_samples": len(combined_df),
        "date_range": {
            "start": combined_df["event_time"].min().isoformat() if "event_time" in combined_df.columns else None,
            "end": combined_df["event_time"].max().isoformat() if "event_time" in combined_df.columns else None,
        }
    }


@task
def train_models_with_comparison(
    data_result: Dict[str, Any],
    model_types: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cross_validation: bool = True
) -> Dict[str, Any]:
    """여러 모델 훈련 및 성능 비교"""
    logger = get_run_logger()

    df = data_result["dataframe"]

    if model_types is None:
        model_types = ["linear", "ridge", "random_forest", "gradient_boosting"]

    logger.info(f"Training {len(model_types)} models: {model_types}")

    # 모델 비교 실행
    training_results = compare_models(
        df=df,
        model_types=model_types,
        test_size=test_size,
        random_state=random_state,
        cross_validation=cross_validation
    )

    if not training_results:
        raise ValueError("No models trained successfully")

    # 최고 성능 모델 선택
    best_model_type, best_result = get_best_model(training_results)

    logger.info(
        f"Best model: {best_model_type} "
        f"(RMSE: {best_result.metrics.rmse:.3f}, "
        f"MAE: {best_result.metrics.mae:.3f}, "
        f"R²: {best_result.metrics.r2:.3f})"
    )

    # 결과 요약
    results_summary = {}
    for model_type, result in training_results.items():
        results_summary[model_type] = {
            "rmse": result.metrics.rmse,
            "mae": result.metrics.mae,
            "r2": result.metrics.r2,
            "cv_mean": result.metrics.cv_mean,
            "cv_std": result.metrics.cv_std,
            "training_samples": result.training_data_size,
            "validation_samples": result.validation_data_size,
        }

    return {
        "training_results": training_results,
        "best_model_type": best_model_type,
        "best_result": best_result,
        "results_summary": results_summary,
    }


@task
def log_experiments_to_mlflow(
    training_result: Dict[str, Any],
    data_result: Dict[str, Any],
    mlflow_manager: MLflowManager
) -> Dict[str, str]:
    """MLflow에 실험 결과 로깅"""
    logger = get_run_logger()

    dataset_info = {
        "total_samples": data_result["total_samples"],
        "quality_score": data_result["quality_score"],
        "date_range_start": data_result["date_range"]["start"],
        "date_range_end": data_result["date_range"]["end"],
    }

    # 모델 비교 결과 로깅
    comparison_name = f"model_comparison_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_ids = mlflow_manager.log_model_comparison(
        results=training_result["training_results"],
        dataset_info=dataset_info,
        comparison_name=comparison_name
    )

    logger.info(f"Logged {len(run_ids)} model runs to MLflow")

    # 최고 성능 모델의 run_id 반환
    best_model_type = training_result["best_model_type"]
    best_run_id = run_ids[best_model_type]

    return {
        "run_ids": run_ids,
        "best_run_id": best_run_id,
        "comparison_name": comparison_name,
    }


@task
def save_models_to_s3(
    training_result: Dict[str, Any],
    mlflow_result: Dict[str, str],
    s3_manager: Optional[S3StorageManager]
) -> Dict[str, Optional[str]]:
    """S3에 모델 저장"""
    logger = get_run_logger()

    if s3_manager is None:
        logger.warning("S3 manager not available, skipping S3 storage")
        return {"best_model_s3_key": None}

    best_model_type = training_result["best_model_type"]
    best_result = training_result["best_result"]
    best_run_id = mlflow_result["best_run_id"]

    # 최고 성능 모델을 staging으로 저장
    try:
        import tempfile
        import joblib

        # 모델 인스턴스 생성
        model = ModelFactory.create_model(best_model_type)
        model.model = best_result.model
        model.scaler = best_result.scaler
        model.feature_names = best_result.feature_names
        model.is_fitted = True

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            model.save_model(f.name)

            with open(f.name, 'rb') as model_file:
                model_data = model_file.read()

        # S3에 저장 (staging 스테이지)
        metadata = {
            "model_type": best_model_type,
            "mlflow_run_id": best_run_id,
            "rmse": str(best_result.metrics.rmse),
            "mae": str(best_result.metrics.mae),
            "r2_score": str(best_result.metrics.r2),
            "features": ",".join(best_result.feature_names),
            "training_size": str(best_result.training_data_size),
            "comparison_experiment": mlflow_result["comparison_name"],
        }

        s3_key = s3_manager.store_model_artifact(
            model_data=model_data,
            model_name="commute-weather",
            version=f"run_{best_run_id}",
            stage="staging",
            metadata=metadata
        )

        # 임시 파일 정리
        os.unlink(f.name)

        logger.info(f"Best model saved to S3 staging: {s3_key}")

        return {"best_model_s3_key": s3_key}

    except Exception as e:
        logger.error(f"Failed to save model to S3: {e}")
        return {"best_model_s3_key": None}


@task
def validate_model_performance(
    training_result: Dict[str, Any],
    baseline_rmse: float = 15.0,
    min_r2: float = 0.5
) -> Dict[str, Any]:
    """모델 성능 검증"""
    logger = get_run_logger()

    best_result = training_result["best_result"]
    metrics = best_result.metrics

    validation_checks = {
        "rmse_acceptable": metrics.rmse <= baseline_rmse,
        "r2_acceptable": metrics.r2 >= min_r2,
        "cross_validation_stable": True,  # 기본값
    }

    # 교차 검증 안정성 확인
    if metrics.cv_mean is not None and metrics.cv_std is not None:
        cv_coefficient_of_variation = metrics.cv_std / metrics.cv_mean
        validation_checks["cross_validation_stable"] = cv_coefficient_of_variation < 0.2

    # 전체 검증 통과 여부
    all_passed = all(validation_checks.values())

    validation_summary = {
        "overall_pass": all_passed,
        "checks": validation_checks,
        "metrics": {
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2": metrics.r2,
            "cv_mean": metrics.cv_mean,
            "cv_std": metrics.cv_std,
        },
        "thresholds": {
            "baseline_rmse": baseline_rmse,
            "min_r2": min_r2,
        }
    }

    if all_passed:
        logger.info("Model performance validation passed")
    else:
        failed_checks = [k for k, v in validation_checks.items() if not v]
        logger.warning(f"Model performance validation failed: {failed_checks}")

    return validation_summary


@flow(name="enhanced-model-training-pipeline")
def enhanced_training_flow(
    *,
    project_root: Optional[Path] = None,
    feature_version: str = "v1",
    model_types: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_samples: int = 24,
    lookback_days: int = 30,
    use_s3: bool = True,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "commute-weather-training",
    baseline_rmse: float = 15.0,
    min_r2: float = 0.5
) -> TrainingArtifact:
    """강화된 모델 훈련 파이프라인"""

    logger = get_run_logger()
    paths = ProjectPaths.from_root(project_root or Path(__file__).resolve().parents[3])

    timestamp = dt.datetime.now(ZoneInfo("Asia/Seoul"))
    logger.info("Starting enhanced model training pipeline", extra={"timestamp": timestamp.isoformat()})

    # 스토리지 설정
    if use_s3:
        s3_bucket = s3_bucket or os.getenv("COMMUTE_S3_BUCKET")
        if not s3_bucket:
            logger.warning("S3 bucket not specified, using local storage only")
            use_s3 = False

    storage = StorageManager(
        paths,
        bucket=s3_bucket if use_s3 else None,
        prefix=s3_prefix,
    )

    s3_manager = None
    if use_s3 and s3_bucket:
        try:
            s3_manager = S3StorageManager(bucket_name=s3_bucket, prefix=s3_prefix)
            logger.info(f"Using S3 storage: {s3_bucket}")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 manager: {e}")
            s3_manager = None

    # MLflow 설정
    mlflow_manager = MLflowManager(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        s3_manager=s3_manager
    )

    # 파이프라인 실행
    data_future = collect_training_data(
        storage,
        feature_version=feature_version,
        min_samples=min_samples,
        lookback_days=lookback_days
    )

    training_future = train_models_with_comparison(
        data_future,
        model_types=model_types,
        test_size=test_size,
        random_state=random_state
    )

    mlflow_future = log_experiments_to_mlflow(
        training_future,
        data_future,
        mlflow_manager
    )

    s3_future = save_models_to_s3(
        training_future,
        mlflow_future,
        s3_manager
    )

    validation_future = validate_model_performance(
        training_future,
        baseline_rmse=baseline_rmse,
        min_r2=min_r2
    )

    # 결과 수집
    data_result = data_future.result()
    training_result = training_future.result()
    mlflow_result = mlflow_future.result()
    s3_result = s3_future.result()
    validation_result = validation_future.result()

    # 아티팩트 생성
    artifact = TrainingArtifact(
        best_model_type=training_result["best_model_type"],
        best_model_s3_key=s3_result["best_model_s3_key"],
        mlflow_run_id=mlflow_result["best_run_id"],
        model_comparison_results=mlflow_result["run_ids"],
        training_data_quality_score=data_result["quality_score"],
        training_samples=training_result["best_result"].training_data_size,
        validation_samples=training_result["best_result"].validation_data_size,
        best_model_metrics={
            "rmse": training_result["best_result"].metrics.rmse,
            "mae": training_result["best_result"].metrics.mae,
            "r2": training_result["best_result"].metrics.r2,
        },
        timestamp=timestamp
    )

    # 성능 검증 결과 로깅
    if not validation_result["overall_pass"]:
        logger.warning("Model performance validation failed")
    else:
        logger.info("Model training completed successfully with validation passed")

    logger.info(
        f"Training pipeline completed: {artifact.best_model_type} "
        f"(RMSE: {artifact.best_model_metrics['rmse']:.3f})"
    )

    return artifact


if __name__ == "__main__":
    # 테스트 실행
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        result = enhanced_training_flow()
        print(f"Training result: {result}")
    else:
        print("Use 'python enhanced_training_pipeline.py test' to run a test")