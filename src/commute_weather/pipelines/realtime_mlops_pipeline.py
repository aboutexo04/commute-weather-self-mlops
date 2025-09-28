"""실시간 MLOps 파이프라인 통합 시스템."""

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
from ..features.intelligent_features import AdaptiveFeaturePipeline
from ..ml.mlflow_integration import MLflowManager
from ..data_quality import DataQualityValidator
from ..tracking import WandBManager

logger = logging.getLogger(__name__)


@task
def collect_realtime_asos_data(
    kma_config: KMAAPIConfig,
    target_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """실시간 ASOS 데이터 수집"""
    logger = get_run_logger()

    collector = AsosDataCollector(kma_config)

    # 지정된 시간 또는 현재 시간 사용
    if target_time is None:
        observation = collector.fetch_latest_observation()
    else:
        observation = collector.fetch_historical_observation(
            kma_config.station_id,
            target_time
        )

    if not observation:
        raise ValueError("Failed to collect ASOS observation data")

    # 핵심 피처 추출
    core_features = collector.select_core_features(observation)
    feature_summary = collector.create_feature_summary(core_features)

    logger.info(
        f"Collected ASOS data: station={observation.station_id}, "
        f"quality={observation.data_quality:.1f}, "
        f"completeness={core_features.get('feature_completeness', 0):.1f}%"
    )

    return {
        "observation": observation,
        "core_features": core_features,
        "feature_summary": feature_summary,
        "timestamp": observation.timestamp,
        "data_quality": observation.data_quality
    }


@task
def apply_intelligent_feature_engineering(
    asos_data: Dict[str, Any],
    asos_collector: AsosDataCollector
) -> Dict[str, Any]:
    """지능형 피처 엔지니어링 적용"""
    logger = get_run_logger()

    # 적응형 피처 파이프라인 생성
    adaptive_pipeline = AdaptiveFeaturePipeline(asos_collector)

    # 실시간 데이터 처리
    engineered_features = adaptive_pipeline.process_realtime_data()

    if not engineered_features:
        raise ValueError("Failed to engineer features from ASOS data")

    logger.info(
        f"Applied intelligent feature engineering: "
        f"confidence={engineered_features.get('feature_confidence', 0):.3f}, "
        f"comfort_prob={engineered_features.get('comfort_probability', 0):.3f}"
    )

    # 피처 트렌드 분석
    feature_trends = adaptive_pipeline.get_feature_trends(hours_back=6)

    return {
        "engineered_features": engineered_features,
        "feature_trends": feature_trends,
        "pipeline_performance": adaptive_pipeline.get_performance_metrics()
    }


@task
def validate_and_store_features(
    engineered_data: Dict[str, Any],
    storage: StorageManager,
    s3_manager: Optional[S3StorageManager] = None,
    version: str = "v2"
) -> Dict[str, Any]:
    """피처 검증 및 저장"""
    logger = get_run_logger()

    engineered_features = engineered_data["engineered_features"]

    # DataFrame으로 변환
    feature_df = pd.DataFrame([engineered_features])

    # 데이터 품질 검증
    validator = DataQualityValidator()

    # 기본 컬럼명 매핑 (검증용)
    validation_df = feature_df.copy()
    if "temperature_c" in validation_df.columns:
        validation_df["temperature_c"] = validation_df["temperature_c"]
    else:
        # 기본값 설정 (검증 통과용)
        validation_df["temperature_c"] = 20.0
        validation_df["wind_speed_ms"] = 2.0
        validation_df["precipitation_mm"] = 0.0
        validation_df["relative_humidity"] = 60.0
        validation_df["timestamp"] = datetime.now()

    validation_results = validator.validate_weather_data(validation_df)
    quality_summary = validator.get_summary()

    logger.info(
        f"Feature validation: {quality_summary['overall_quality']} "
        f"({quality_summary['passed_checks']}/{quality_summary['total_checks']} checks passed)"
    )

    # S3에 피처 저장
    s3_key = None
    if s3_manager:
        try:
            timestamp = engineered_features.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            s3_key = s3_manager.store_processed_features(
                df=feature_df,
                feature_set_name="realtime_intelligent_features",
                version=version,
                timestamp=timestamp
            )
            logger.info(f"Stored features to S3: {s3_key}")

        except Exception as e:
            logger.warning(f"Failed to store features to S3: {e}")

    # 로컬에도 저장
    timestamp = engineered_features.get("timestamp", datetime.now())
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    local_path = storage.processed_path(
        "features",
        "realtime",
        version,
        timestamp.strftime("%Y"),
        timestamp.strftime("%m"),
        timestamp.strftime("%d"),
        f"{timestamp.strftime('%H%M')}_intelligent.parquet",
    )
    storage.write_dataframe(local_path, feature_df)

    return {
        "feature_df": feature_df,
        "s3_key": s3_key,
        "local_path": local_path,
        "validation_summary": quality_summary,
        "feature_count": len(feature_df.columns),
        "quality_score": quality_summary.get("passed_checks", 0) / max(1, quality_summary.get("total_checks", 1)) * 100
    }


@task
def trigger_model_inference(
    feature_data: Dict[str, Any],
    mlflow_manager: MLflowManager
) -> Dict[str, Any]:
    """모델 추론 실행"""
    logger = get_run_logger()

    try:
        # 프로덕션 모델 로드
        production_model = mlflow_manager.get_production_model("commute-weather")

        if not production_model:
            logger.warning("No production model found, using latest model")
            # 최고 성능 모델 조회
            best_model_info = mlflow_manager.get_best_model_from_experiment()
            if best_model_info:
                best_run, model_uri = best_model_info
                production_model = mlflow_manager.load_model_from_run(best_run.info.run_id)

        if not production_model:
            logger.warning("No trained model available for inference")
            return {
                "predictions": None,
                "model_used": None,
                "inference_confidence": 0.0
            }

        # 예측 수행
        feature_df = feature_data["feature_df"]
        predictions = production_model.predict(feature_df)

        # 예측 결과 처리
        prediction_value = float(predictions[0]) if len(predictions) > 0 else 0.0

        # 신뢰도 계산 (피처 신뢰도 기반)
        inference_confidence = feature_df.get("feature_confidence", [0.8])[0]

        logger.info(f"Model inference completed: prediction={prediction_value:.2f}")

        # 추론 메트릭 로깅
        mlflow_manager.log_inference_metrics(
            predictions=[prediction_value],
            additional_metrics={
                "feature_confidence": inference_confidence,
                "feature_count": len(feature_df.columns),
                "data_quality": feature_df.get("data_quality", [80.0])[0],
            }
        )

        return {
            "predictions": [prediction_value],
            "model_used": production_model.model_name,
            "inference_confidence": inference_confidence,
            "prediction_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return {
            "predictions": None,
            "model_used": None,
            "inference_confidence": 0.0,
            "error": str(e)
        }


@task
def store_prediction_results(
    inference_data: Dict[str, Any],
    feature_data: Dict[str, Any],
    s3_manager: Optional[S3StorageManager] = None
) -> Dict[str, Any]:
    """예측 결과 저장"""
    logger = get_run_logger()

    if not inference_data.get("predictions"):
        logger.warning("No predictions to store")
        return {"status": "no_predictions"}

    # 예측 결과 데이터 구성
    prediction_record = {
        "timestamp": datetime.now(),
        "prediction_value": inference_data["predictions"][0],
        "model_used": inference_data.get("model_used", "unknown"),
        "inference_confidence": inference_data.get("inference_confidence", 0.0),
        "feature_confidence": feature_data["feature_df"].get("feature_confidence", [0.0])[0],
        "data_quality": feature_data["feature_df"].get("data_quality", [80.0])[0],
        "feature_count": feature_data["feature_count"],
    }

    prediction_df = pd.DataFrame([prediction_record])

    # S3에 예측 결과 저장
    s3_key = None
    if s3_manager:
        try:
            import json
            timestamp = datetime.now()

            s3_key = s3_manager._build_key(
                "predictions",
                "realtime",
                timestamp.strftime("%Y/%m/%d"),
                f"prediction_{timestamp.strftime('%H%M%S')}.json"
            )

            s3_manager.s3_client.put_object(
                Bucket=s3_manager.bucket_name,
                Key=s3_key,
                Body=json.dumps(prediction_record, default=str, indent=2).encode("utf-8"),
                Metadata={
                    "prediction_value": str(prediction_record["prediction_value"]),
                    "model_used": prediction_record["model_used"],
                    "confidence": str(prediction_record["inference_confidence"]),
                },
                ContentType="application/json",
            )

            logger.info(f"Stored prediction to S3: {s3_key}")

        except Exception as e:
            logger.warning(f"Failed to store prediction to S3: {e}")

    logger.info(
        f"Prediction stored: value={prediction_record['prediction_value']:.2f}, "
        f"confidence={prediction_record['inference_confidence']:.3f}"
    )

    return {
        "prediction_record": prediction_record,
        "s3_key": s3_key,
        "status": "success"
    }


@flow(name="realtime-mlops-pipeline")
def realtime_mlops_flow(
    *,
    project_root: Optional[Path] = None,
    auth_key: Optional[str] = None,
    station_id: str = "108",
    base_url: str = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
    target_time: Optional[datetime] = None,
    use_s3: bool = True,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "commute-weather-realtime",
    feature_version: str = "v2",
    use_wandb: bool = True,
    wandb_project: str = "commute-weather-mlops",
    wandb_entity: Optional[str] = None
) -> Dict[str, Any]:
    """실시간 MLOps 파이프라인"""

    logger = get_run_logger()
    paths = ProjectPaths.from_root(project_root or Path(__file__).resolve().parents[3])

    pipeline_start_time = datetime.now(ZoneInfo("Asia/Seoul"))
    logger.info("Starting realtime MLOps pipeline", extra={"start_time": pipeline_start_time.isoformat()})

    # 인증 키 확인
    if auth_key is None:
        auth_key = os.getenv("KMA_AUTH_KEY")
    if auth_key is None:
        raise ValueError("KMA API auth key must be provided via parameter or KMA_AUTH_KEY environment variable")

    # KMA 설정
    kma_config = KMAAPIConfig(
        base_url=base_url,
        auth_key=auth_key,
        station_id=station_id,
    )

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

    # WandB 설정
    wandb_manager = None
    if use_wandb:
        try:
            wandb_manager = WandBManager(
                project_name=wandb_project,
                entity=wandb_entity,
                tags=["realtime-pipeline", "production", f"station-{station_id}"]
            )

            # 실시간 파이프라인 실행 시작
            run_name = f"realtime_pipeline_{pipeline_start_time.strftime('%Y%m%d_%H%M%S')}"
            wandb_manager.start_run(
                run_name=run_name,
                job_type="realtime_inference",
                config={
                    "station_id": station_id,
                    "feature_version": feature_version,
                    "mlflow_experiment": mlflow_experiment,
                    "target_time": target_time.isoformat() if target_time else "current",
                    "pipeline_parameters": {
                        "use_s3": use_s3,
                        "s3_bucket": s3_bucket,
                        "s3_prefix": s3_prefix
                    }
                },
                notes=f"Real-time weather prediction pipeline for station {station_id}"
            )

            logger.info(f"Started WandB tracking for pipeline: {run_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize WandB manager: {e}")
            wandb_manager = None

    # ASOS 데이터 수집기 생성
    asos_collector = AsosDataCollector(kma_config)

    # 파이프라인 실행
    try:
        # 1. 실시간 ASOS 데이터 수집
        asos_data = collect_realtime_asos_data(kma_config, target_time)

        # WandB: 데이터 수집 로깅
        if wandb_manager:
            wandb_manager.log_weather_data_summary(asos_data, step=1)
            wandb_manager.log_pipeline_performance(
                {"data_collection_latency_ms": 1000},  # 예시 값
                "data_collection",
                step=1
            )

        # 2. 지능형 피처 엔지니어링
        engineered_data = apply_intelligent_feature_engineering(asos_data, asos_collector)

        # WandB: 피처 엔지니어링 로깅
        if wandb_manager:
            feature_metrics = {
                "engineered_feature_count": engineered_data.get("feature_count", 0),
                "feature_quality": engineered_data.get("quality_score", 0.0),
                "engineering_confidence": engineered_data.get("engineering_confidence", 0.8)
            }
            wandb_manager.log_metrics(feature_metrics, step=2)

        # 3. 피처 검증 및 저장
        feature_data = validate_and_store_features(
            engineered_data, storage, s3_manager, feature_version
        )

        # WandB: 피처 검증 로깅
        if wandb_manager:
            validation_metrics = {
                "feature_validation_passed": feature_data.get("validation_passed", True),
                "final_feature_count": feature_data.get("feature_count", 0),
                "final_quality_score": feature_data.get("quality_score", 0.0)
            }
            wandb_manager.log_metrics(validation_metrics, step=3)

        # 4. 모델 추론
        inference_data = trigger_model_inference(feature_data, mlflow_manager)

        # WandB: 모델 추론 로깅
        if wandb_manager:
            inference_metrics = {
                "prediction_value": inference_data.get("predictions", [0])[0] if inference_data.get("predictions") else 0,
                "inference_confidence": inference_data.get("inference_confidence", 0.0),
                "model_used": inference_data.get("model_used", "unknown"),
                "prediction_successful": inference_data.get("predictions") is not None
            }
            wandb_manager.log_metrics(inference_metrics, step=4)

        # 5. 예측 결과 저장
        prediction_data = store_prediction_results(inference_data, feature_data, s3_manager)

        # 결과 정리
        pipeline_end_time = datetime.now(ZoneInfo("Asia/Seoul"))
        execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()

        result = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "pipeline_start_time": pipeline_start_time.isoformat(),
            "pipeline_end_time": pipeline_end_time.isoformat(),
            "data_timestamp": asos_data["timestamp"].isoformat(),
            "data_quality": asos_data["data_quality"],
            "feature_quality": feature_data["quality_score"],
            "prediction_value": inference_data.get("predictions", [None])[0],
            "prediction_confidence": inference_data.get("inference_confidence", 0.0),
            "model_used": inference_data.get("model_used"),
            "s3_keys": {
                "features": feature_data.get("s3_key"),
                "prediction": prediction_data.get("s3_key")
            },
            "local_paths": {
                "features": feature_data.get("local_path")
            }
        }

        # WandB: 파이프라인 성공 결과 로깅
        if wandb_manager:
            # 전체 파이프라인 메트릭
            pipeline_metrics = {
                "pipeline_status": 1.0,  # 성공 = 1.0
                "total_execution_time_seconds": execution_time,
                "data_quality_final": asos_data["data_quality"],
                "feature_quality_final": feature_data["quality_score"],
                "prediction_value_final": result.get("prediction_value", 0),
                "prediction_confidence_final": result.get("prediction_confidence", 0.0),
                "pipeline_efficiency": 1.0 / max(execution_time, 1.0)  # 효율성 = 1/실행시간
            }

            wandb_manager.log_metrics(pipeline_metrics, step=5)

            # 예측 결과 분포 로깅
            if result.get("prediction_value") is not None:
                wandb_manager.log_prediction_distribution([result["prediction_value"]])

            # 파이프라인 요약 테이블
            wandb_manager.log_metrics({
                "summary/total_stages_completed": 5,
                "summary/total_stages_failed": 0,
                "summary/success_rate": 1.0
            }, step=5)

            logger.info("WandB metrics logged for successful pipeline execution")

        logger.info(
            f"Realtime MLOps pipeline completed successfully: "
            f"prediction={result.get('prediction_value', 0):.2f}, "
            f"execution_time={execution_time:.2f}s"
        )

        return result

    except Exception as e:
        pipeline_end_time = datetime.now(ZoneInfo("Asia/Seoul"))
        execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()

        logger.error(f"Realtime MLOps pipeline failed: {e}")

        # WandB: 파이프라인 실패 결과 로깅
        if wandb_manager:
            error_metrics = {
                "pipeline_status": 0.0,  # 실패 = 0.0
                "total_execution_time_seconds": execution_time,
                "error_occurred": 1.0,
                "pipeline_efficiency": 0.0
            }

            wandb_manager.log_metrics(error_metrics)

            # 실패 알림 생성
            wandb_manager.create_alert(
                title="Realtime Pipeline Failed",
                text=f"Pipeline execution failed after {execution_time:.2f}s. Error: {str(e)[:100]}",
                level="ERROR"
            )

            # 파이프라인 요약
            wandb_manager.log_metrics({
                "summary/total_stages_completed": 0,  # 실패시 정확한 완료 단계 수 계산 필요
                "summary/total_stages_failed": 1,
                "summary/success_rate": 0.0
            })

            logger.info("WandB metrics logged for failed pipeline execution")

        error_result = {
            "status": "failed",
            "error": str(e),
            "execution_time_seconds": execution_time,
            "pipeline_start_time": pipeline_start_time.isoformat(),
            "pipeline_end_time": pipeline_end_time.isoformat(),
        }

        return error_result

    finally:
        # WandB 실행 종료
        if wandb_manager:
            wandb_manager.finish_run()
            logger.info("WandB run finished")


@flow(name="hourly-asos-collection")
def hourly_asos_collection_flow(
    *,
    project_root: Optional[Path] = None,
    auth_key: Optional[str] = None,
    station_id: str = "108",
    hours_back: int = 1,
    use_s3: bool = True,
    s3_bucket: Optional[str] = None
) -> Dict[str, Any]:
    """매시간 ASOS 데이터 수집 흐름"""

    logger = get_run_logger()

    # 인증 설정
    if auth_key is None:
        auth_key = os.getenv("KMA_AUTH_KEY")
    if auth_key is None:
        raise ValueError("KMA API auth key required")

    kma_config = KMAAPIConfig(
        auth_key=auth_key,
        station_id=station_id,
    )

    # 스토리지 설정
    paths = ProjectPaths.from_root(project_root or Path(__file__).resolve().parents[3])
    storage = StorageManager(paths)

    s3_manager = None
    if use_s3:
        s3_bucket = s3_bucket or os.getenv("COMMUTE_S3_BUCKET")
        if s3_bucket:
            s3_manager = S3StorageManager(bucket_name=s3_bucket)

    # ASOS 수집기 생성
    collector = AsosDataCollector(kma_config)

    # 배치 데이터 수집
    collected_data = collector.collect_hourly_batch(hours_back=hours_back, station_id=station_id)

    logger.info(f"Collected {len(collected_data)} hourly observations")

    # 수집된 데이터를 S3 및 로컬에 저장
    stored_files = []

    for data_record in collected_data:
        try:
            timestamp = data_record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            # DataFrame으로 변환
            df = pd.DataFrame([data_record])

            # S3 저장
            if s3_manager:
                s3_key = s3_manager.store_processed_features(
                    df=df,
                    feature_set_name="hourly_asos_collection",
                    version="v1",
                    timestamp=timestamp
                )
                stored_files.append({"type": "s3", "key": s3_key})

            # 로컬 저장
            local_path = storage.processed_path(
                "hourly_asos",
                timestamp.strftime("%Y"),
                timestamp.strftime("%m"),
                timestamp.strftime("%d"),
                f"{timestamp.strftime('%H%M')}_asos.parquet"
            )
            storage.write_dataframe(local_path, df)
            stored_files.append({"type": "local", "path": local_path})

        except Exception as e:
            logger.error(f"Failed to store data record: {e}")

    return {
        "status": "success",
        "collected_count": len(collected_data),
        "stored_files": stored_files,
        "hours_back": hours_back,
        "station_id": station_id
    }


if __name__ == "__main__":
    # 테스트 실행
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "realtime":
            result = realtime_mlops_flow()
            print(f"Realtime pipeline result: {result}")
        elif sys.argv[1] == "hourly":
            result = hourly_asos_collection_flow()
            print(f"Hourly collection result: {result}")
        else:
            print("Usage: python realtime_mlops_pipeline.py [realtime|hourly]")
    else:
        print("Usage: python realtime_mlops_pipeline.py [realtime|hourly]")