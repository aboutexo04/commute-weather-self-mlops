"""모델 평가 및 검증 파이프라인."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from prefect import flow, task, get_run_logger
from zoneinfo import ZoneInfo

from ..config import KMAAPIConfig, ProjectPaths
from ..storage import StorageManager
from ..storage.s3_manager import S3StorageManager
from ..data_sources.asos_collector import AsosDataCollector
from ..ml.mlflow_integration import MLflowManager
from ..evaluation import ModelEvaluator, ModelPerformanceMetrics, ValidationResult
from ..tracking import WandBManager

logger = logging.getLogger(__name__)


@task
def collect_evaluation_data(
    kma_config: KMAAPIConfig,
    hours_back: int = 24,
    min_data_points: int = 20
) -> Dict[str, Any]:
    """평가용 데이터 수집."""
    logger = get_run_logger()

    collector = AsosDataCollector(kma_config)

    # 과거 데이터 수집
    collected_data = collector.collect_hourly_batch(
        hours_back=hours_back,
        station_id=kma_config.station_id
    )

    if len(collected_data) < min_data_points:
        raise ValueError(
            f"Insufficient data for evaluation: got {len(collected_data)}, "
            f"need at least {min_data_points}"
        )

    logger.info(f"Collected {len(collected_data)} data points for evaluation")

    # 데이터를 DataFrame으로 변환
    eval_df = pd.DataFrame(collected_data)

    # 기본 통계
    data_stats = {
        "total_records": len(eval_df),
        "date_range": {
            "start": eval_df["timestamp"].min().isoformat(),
            "end": eval_df["timestamp"].max().isoformat()
        },
        "data_quality_avg": eval_df["data_quality"].mean(),
        "data_quality_std": eval_df["data_quality"].std(),
        "feature_completeness_avg": eval_df["feature_completeness"].mean()
    }

    return {
        "evaluation_data": collected_data,
        "evaluation_df": eval_df,
        "data_statistics": data_stats,
        "collection_timestamp": datetime.now(ZoneInfo("Asia/Seoul"))
    }


@task
def generate_predictions_for_evaluation(
    evaluation_data: Dict[str, Any],
    mlflow_manager: MLflowManager,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """평가용 예측 생성."""
    logger = get_run_logger()

    eval_df = evaluation_data["evaluation_df"]

    # 모델 로드
    if model_name:
        production_model = mlflow_manager.load_production_model(model_name)
    else:
        # 가장 최근의 프로덕션 모델 로드
        best_model_info = mlflow_manager.get_best_model()
        if best_model_info:
            best_run, model_uri = best_model_info
            production_model = mlflow_manager.load_model_from_run(best_run.info.run_id)
        else:
            raise ValueError("No production model available for evaluation")

    if not production_model:
        raise ValueError(f"Failed to load model: {model_name or 'best_model'}")

    logger.info(f"Using model for evaluation: {production_model.model_name}")

    # 피처 준비 (실제 피처 컬럼만 추출)
    feature_columns = [col for col in eval_df.columns
                      if col.endswith(('_c', '_pct', '_mm_1h', '_ms', '_hpa', '_km', '_deg', '_tenths'))]

    if not feature_columns:
        raise ValueError("No feature columns found in evaluation data")

    feature_df = eval_df[feature_columns].fillna(0)

    # 예측 생성
    predictions = []
    prediction_metadata = []

    for idx, row in eval_df.iterrows():
        try:
            # 단일 예측 수행
            single_feature = feature_df.iloc[idx:idx+1]
            prediction = production_model.predict(single_feature)

            prediction_value = float(prediction[0]) if len(prediction) > 0 else 0.0
            predictions.append(prediction_value)

            # 메타데이터 수집
            metadata = {
                "timestamp": row["timestamp"],
                "data_quality": row["data_quality"],
                "feature_completeness": row.get("feature_completeness", 0.0),
                "confidence": min(row["data_quality"] / 100.0, 1.0),
                "feature_quality": row.get("feature_reliability", 100.0),
                "prediction_latency_ms": 50.0,  # 예시 값
                "model_size_mb": 1.5  # 예시 값
            }
            prediction_metadata.append(metadata)

        except Exception as e:
            logger.warning(f"Failed to generate prediction for row {idx}: {e}")
            predictions.append(0.0)
            prediction_metadata.append({
                "timestamp": row["timestamp"],
                "data_quality": 0.0,
                "confidence": 0.0,
                "error": str(e)
            })

    logger.info(f"Generated {len(predictions)} predictions for evaluation")

    return {
        "predictions": predictions,
        "prediction_metadata": prediction_metadata,
        "model_name": production_model.model_name,
        "model_version": getattr(production_model, 'model_version', 'unknown'),
        "feature_columns": feature_columns,
        "evaluation_period_hours": len(predictions)
    }


@task
def calculate_model_performance(
    evaluation_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    model_evaluator: ModelEvaluator
) -> Dict[str, Any]:
    """모델 성능 계산."""
    logger = get_run_logger()

    eval_df = evaluation_data["evaluation_df"]
    predictions = prediction_data["predictions"]
    prediction_metadata = prediction_data["prediction_metadata"]

    # 실제값 생성 (시뮬레이션 - 실제 환경에서는 실제 관측값 사용)
    # 여기서는 온도를 기반으로 한 간단한 함수로 실제값 시뮬레이션
    actual_values = []
    for _, row in eval_df.iterrows():
        temp = row.get("temperature_c", 20.0)
        precip = row.get("precipitation_mm_1h", 0.0)

        # 간단한 경험적 공식 (실제 환경에서는 실제 측정값 사용)
        actual_value = (temp * 0.5) + (precip * 0.3) + 10.0  # 예시 공식
        actual_values.append(actual_value)

    if len(predictions) != len(actual_values):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != actual values count ({len(actual_values)})"
        )

    # 성능 메트릭 계산
    performance_metrics = model_evaluator.evaluate_model_performance(
        model_name=prediction_data["model_name"],
        model_version=prediction_data["model_version"],
        predictions=predictions,
        actual_values=actual_values,
        prediction_metadata=prediction_metadata,
        evaluation_period_hours=prediction_data["evaluation_period_hours"]
    )

    logger.info(
        f"Model performance calculated: "
        f"R²={performance_metrics.r2:.3f}, "
        f"MAE={performance_metrics.mae:.3f}, "
        f"MAPE={performance_metrics.mape:.1f}%"
    )

    return {
        "performance_metrics": performance_metrics,
        "actual_values": actual_values,
        "predictions": predictions,
        "evaluation_summary": {
            "r2_score": performance_metrics.r2,
            "mae": performance_metrics.mae,
            "rmse": performance_metrics.rmse,
            "mape": performance_metrics.mape,
            "data_quality": performance_metrics.data_quality_score,
            "prediction_confidence": performance_metrics.prediction_confidence
        }
    }


@task
def validate_model_performance(
    performance_data: Dict[str, Any],
    model_evaluator: ModelEvaluator
) -> Dict[str, Any]:
    """모델 성능 검증."""
    logger = get_run_logger()

    performance_metrics = performance_data["performance_metrics"]

    # 과거 성능 메트릭 조회
    historical_metrics = model_evaluator.get_historical_metrics(
        performance_metrics.model_name,
        days_back=7
    )

    # 모델 검증 수행
    validation_result = model_evaluator.validate_model(
        performance_metrics,
        historical_metrics
    )

    logger.info(
        f"Model validation completed: "
        f"valid={validation_result.is_valid}, "
        f"confidence={validation_result.confidence_score:.3f}, "
        f"retraining_required={validation_result.retraining_required}"
    )

    # 성능 보고서 생성
    performance_report = model_evaluator.generate_performance_report(
        validation_result,
        historical_metrics
    )

    return {
        "validation_result": validation_result,
        "performance_report": performance_report,
        "historical_metrics_count": len(historical_metrics),
        "validation_summary": {
            "is_valid": validation_result.is_valid,
            "confidence_score": validation_result.confidence_score,
            "retraining_required": validation_result.retraining_required,
            "issues_count": len(validation_result.validation_issues),
            "recommendations_count": len(validation_result.recommendations)
        }
    }


@flow(name="model-evaluation-pipeline")
def model_evaluation_flow(
    *,
    project_root: Optional[Path] = None,
    auth_key: Optional[str] = None,
    station_id: str = "108",
    model_name: Optional[str] = None,
    hours_back: int = 24,
    min_data_points: int = 20,
    use_s3: bool = True,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "commute-weather-evaluation",
    use_wandb: bool = True,
    wandb_project: str = "commute-weather-mlops",
    wandb_entity: Optional[str] = None
) -> Dict[str, Any]:
    """모델 평가 파이프라인."""

    logger = get_run_logger()
    paths = ProjectPaths.from_root(project_root or Path(__file__).resolve().parents[3])

    evaluation_start_time = datetime.now(ZoneInfo("Asia/Seoul"))
    logger.info("Starting model evaluation pipeline", extra={"start_time": evaluation_start_time.isoformat()})

    # 인증 키 확인
    if auth_key is None:
        auth_key = os.getenv("KMA_AUTH_KEY")
    if auth_key is None:
        raise ValueError("KMA API auth key must be provided")

    # 설정 초기화
    kma_config = KMAAPIConfig(
        auth_key=auth_key,
        station_id=station_id,
    )

    # S3 스토리지 설정
    s3_manager = None
    if use_s3:
        s3_bucket = s3_bucket or os.getenv("COMMUTE_S3_BUCKET")
        if s3_bucket:
            s3_manager = S3StorageManager(bucket_name=s3_bucket, prefix=s3_prefix)

    # MLflow 설정
    mlflow_manager = MLflowManager(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
        s3_manager=s3_manager
    )

    # WandB 설정
    wandb_manager = None
    if use_wandb:
        try:
            wandb_manager = WandBManager(
                project_name=wandb_project,
                entity=wandb_entity,
                tags=["evaluation", "model-validation", f"station-{station_id}"]
            )

            # 평가 실행 시작
            run_name = f"model_evaluation_{evaluation_start_time.strftime('%Y%m%d_%H%M%S')}"
            wandb_manager.start_run(
                run_name=run_name,
                job_type="model_evaluation",
                config={
                    "station_id": station_id,
                    "model_name": model_name or "auto_select",
                    "hours_back": hours_back,
                    "min_data_points": min_data_points,
                    "evaluation_period": f"{hours_back}h",
                    "mlflow_experiment": mlflow_experiment
                },
                notes=f"Model evaluation for station {station_id} with {hours_back}h data"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize WandB manager: {e}")
            wandb_manager = None

    # 모델 평가기 설정
    model_evaluator = ModelEvaluator(
        s3_manager=s3_manager,
        mlflow_tracking_uri=mlflow_tracking_uri,
        wandb_manager=wandb_manager
    )

    try:
        # 1. 평가용 데이터 수집
        evaluation_data = collect_evaluation_data(kma_config, hours_back, min_data_points)

        # WandB: 데이터 수집 로깅
        if wandb_manager:
            data_metrics = {
                "evaluation_data_points": evaluation_data["data_statistics"]["total_records"],
                "avg_data_quality": evaluation_data["data_statistics"]["data_quality_avg"],
                "data_quality_std": evaluation_data["data_statistics"]["data_quality_std"],
                "avg_feature_completeness": evaluation_data["data_statistics"]["feature_completeness_avg"]
            }
            wandb_manager.log_metrics(data_metrics, step=1)

        # 2. 모델 예측 생성
        prediction_data = generate_predictions_for_evaluation(evaluation_data, mlflow_manager, model_name)

        # WandB: 예측 생성 로깅
        if wandb_manager:
            prediction_metrics = {
                "model_used": prediction_data["model_name"],
                "total_predictions": len(prediction_data["predictions"]),
                "feature_columns_count": len(prediction_data["feature_columns"]),
                "avg_prediction_value": sum(prediction_data["predictions"]) / len(prediction_data["predictions"])
            }
            wandb_manager.log_metrics(prediction_metrics, step=2)

            # 예측 분포 로깅
            wandb_manager.log_prediction_distribution(prediction_data["predictions"])

        # 3. 성능 계산
        performance_data = calculate_model_performance(evaluation_data, prediction_data, model_evaluator)

        # WandB: 성능 메트릭 로깅
        if wandb_manager:
            wandb_manager.log_model_performance(
                performance_data["evaluation_summary"],
                prediction_data["model_name"],
                prediction_data["model_version"],
                step=3
            )

        # 4. 모델 검증
        validation_data = validate_model_performance(performance_data, model_evaluator)

        # WandB: 검증 결과 로깅
        if wandb_manager:
            validation_metrics = {
                "model_is_valid": validation_data["validation_summary"]["is_valid"],
                "validation_confidence": validation_data["validation_summary"]["confidence_score"],
                "retraining_required": validation_data["validation_summary"]["retraining_required"],
                "validation_issues": validation_data["validation_summary"]["issues_count"],
                "recommendations": validation_data["validation_summary"]["recommendations_count"]
            }
            wandb_manager.log_metrics(validation_metrics, step=4)

        # 5. 평가 결과 저장
        evaluation_results_key = model_evaluator.store_evaluation_results(
            validation_data["validation_result"],
            experiment_name=mlflow_experiment,
            wandb_run_name=f"evaluation_{prediction_data['model_name']}"
        )

        # 결과 정리
        evaluation_end_time = datetime.now(ZoneInfo("Asia/Seoul"))
        execution_time = (evaluation_end_time - evaluation_start_time).total_seconds()

        result = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "evaluation_start_time": evaluation_start_time.isoformat(),
            "evaluation_end_time": evaluation_end_time.isoformat(),
            "model_name": prediction_data["model_name"],
            "model_version": prediction_data["model_version"],
            "data_points_evaluated": evaluation_data["data_statistics"]["total_records"],
            "performance_metrics": performance_data["evaluation_summary"],
            "validation_results": validation_data["validation_summary"],
            "evaluation_results_key": evaluation_results_key,
            "retraining_required": validation_data["validation_result"].retraining_required,
            "recommendations": validation_data["validation_result"].recommendations
        }

        # WandB: 최종 결과 로깅
        if wandb_manager:
            final_metrics = {
                "evaluation_status": 1.0,  # 성공
                "total_execution_time": execution_time,
                "evaluation_efficiency": evaluation_data["data_statistics"]["total_records"] / max(execution_time, 1.0),
                "overall_model_health": validation_data["validation_summary"]["confidence_score"]
            }
            wandb_manager.log_metrics(final_metrics, step=5)

            # 알림 (필요시)
            if validation_data["validation_result"].retraining_required:
                wandb_manager.create_alert(
                    title="Model Retraining Required",
                    text=f"Model {prediction_data['model_name']} requires retraining based on evaluation results",
                    level="WARN"
                )

        logger.info(
            f"Model evaluation pipeline completed successfully: "
            f"model={prediction_data['model_name']}, "
            f"valid={validation_data['validation_result'].is_valid}, "
            f"confidence={validation_data['validation_result'].confidence_score:.3f}"
        )

        return result

    except Exception as e:
        evaluation_end_time = datetime.now(ZoneInfo("Asia/Seoul"))
        execution_time = (evaluation_end_time - evaluation_start_time).total_seconds()

        logger.error(f"Model evaluation pipeline failed: {e}")

        # WandB: 실패 로깅
        if wandb_manager:
            error_metrics = {
                "evaluation_status": 0.0,  # 실패
                "total_execution_time": execution_time,
                "error_occurred": 1.0
            }
            wandb_manager.log_metrics(error_metrics)

            wandb_manager.create_alert(
                title="Model Evaluation Failed",
                text=f"Evaluation pipeline failed after {execution_time:.2f}s. Error: {str(e)[:100]}",
                level="ERROR"
            )

        return {
            "status": "failed",
            "error": str(e),
            "execution_time_seconds": execution_time,
            "evaluation_start_time": evaluation_start_time.isoformat(),
            "evaluation_end_time": evaluation_end_time.isoformat(),
        }

    finally:
        # WandB 실행 종료
        if wandb_manager:
            wandb_manager.finish_run()
            logger.info("WandB evaluation run finished")


if __name__ == "__main__":
    # 테스트 실행
    result = model_evaluation_flow(hours_back=6, min_data_points=5)
    print(f"Evaluation result: {result}")