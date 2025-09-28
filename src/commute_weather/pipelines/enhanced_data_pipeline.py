"""강화된 데이터 파이프라인 with 품질 검증 및 S3 통합."""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from prefect import flow, task, get_run_logger
from zoneinfo import ZoneInfo

from ..config import KMAAPIConfig, ProjectPaths
from ..data_sources.kma_api import normalize_kma_observations
from ..storage import StorageManager
from ..storage.s3_manager import S3StorageManager
from ..data_quality import DataQualityValidator, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancedDataArtifact:
    """강화된 데이터 아티팩트 메타데이터"""
    raw_data_key: str
    processed_data_key: str
    quality_report_key: str
    feature_data_key: str
    timestamp: dt.datetime
    station_id: str
    data_quality_score: float
    validation_passed: bool
    row_count: int


@task
def ingest_weather_data_with_quality(
    kma_config: KMAAPIConfig,
    storage: StorageManager,
    s3_manager: Optional[S3StorageManager] = None,
    *,
    run_time: Optional[dt.datetime] = None,
    lookback_hours: int = 3,
) -> Dict[str, Any]:
    """품질 검증이 포함된 기상 데이터 수집"""
    logger = get_run_logger()

    tz = ZoneInfo(kma_config.timezone)
    effective_time = (run_time or dt.datetime.now(tz)).astimezone(tz)
    end_time = effective_time.replace(minute=0, second=0, microsecond=0)
    if end_time > effective_time:
        end_time -= dt.timedelta(hours=1)

    span = max(1, lookback_hours)
    start_time = end_time - dt.timedelta(hours=span - 1)

    logger.info(f"Fetching weather data for {start_time} to {end_time}")

    # KMA API 호출
    import requests
    params = {
        "tm1": start_time.strftime("%Y%m%d%H%M"),
        "tm2": end_time.strftime("%Y%m%d%H%M"),
        "stn": kma_config.station_id,
        "help": kma_config.help_flag,
        "authKey": kma_config.auth_key,
    }

    try:
        session = requests.Session()
        response = session.get(kma_config.base_url, params=params, timeout=kma_config.timeout_seconds)
        response.raise_for_status()
        payload_text = response.text
        fetched_at = dt.datetime.now(dt.timezone.utc)

        logger.info(f"Successfully fetched {len(payload_text)} bytes of data")

    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        raise

    # S3에 원시 데이터 저장
    raw_data_key = None
    if s3_manager:
        try:
            raw_data_key = s3_manager.store_raw_weather_data(
                data=payload_text,
                station_id=kma_config.station_id,
                timestamp=end_time,
                source="kma",
                metadata={
                    "lookback_hours": lookback_hours,
                    "request_params": str(params),
                    "response_size": len(payload_text.encode("utf-8"))
                }
            )
            logger.info(f"Stored raw data to S3: {raw_data_key}")
        except Exception as e:
            logger.warning(f"Failed to store raw data to S3: {e}")

    # 로컬 스토리지에도 저장 (백업)
    filename = f"{end_time.strftime('%H%M')}"
    base_parts = (
        "kma",
        kma_config.station_id,
        start_time.strftime("%Y"),
        start_time.strftime("%m"),
        start_time.strftime("%d"),
    )
    local_raw_path = storage.raw_path(*base_parts, f"{filename}.txt")
    storage.write_text(local_raw_path, payload_text)

    return {
        "payload_text": payload_text,
        "raw_data_key": raw_data_key,
        "local_raw_path": local_raw_path,
        "station_id": kma_config.station_id,
        "start_time": start_time,
        "end_time": end_time,
        "fetched_at": fetched_at,
        "response_size": len(payload_text.encode("utf-8"))
    }


@task
def process_and_validate_data(
    ingest_result: Dict[str, Any],
    storage: StorageManager,
    s3_manager: Optional[S3StorageManager] = None,
    *,
    timezone: str = "Asia/Seoul"
) -> Dict[str, Any]:
    """데이터 처리 및 품질 검증"""
    logger = get_run_logger()

    # 데이터 파싱
    try:
        observations = normalize_kma_observations(ingest_result["payload_text"])
        logger.info(f"Parsed {len(observations)} observations")

        if not observations:
            raise ValueError("No observations parsed from payload")

    except Exception as e:
        logger.error(f"Failed to parse observations: {e}")
        raise

    # DataFrame 생성
    rows = []
    for obs in observations:
        timestamp = obs.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo(timezone))
        else:
            timestamp = timestamp.astimezone(ZoneInfo(timezone))

        rows.append({
            "timestamp": timestamp,
            "station_id": ingest_result["station_id"],
            "temperature_c": obs.temperature_c,
            "wind_speed_ms": obs.wind_speed_ms,
            "precipitation_mm": obs.precipitation_mm,
            "relative_humidity": obs.relative_humidity,
            "precipitation_type": obs.precipitation_type or "unknown",
            "source": "kma",
            "ingested_at": ingest_result["fetched_at"],
        })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)

    logger.info(f"Created DataFrame with {len(df)} rows")

    # 데이터 품질 검증
    validator = DataQualityValidator()
    validation_results = validator.validate_weather_data(df)
    quality_summary = validator.get_summary()
    quality_report = validator.to_dict()

    logger.info(
        f"Data quality validation: {quality_summary['overall_quality']} "
        f"({quality_summary['passed_checks']}/{quality_summary['total_checks']} checks passed)"
    )

    # 품질 점수 계산 (0-100)
    total_checks = quality_summary['total_checks']
    passed_checks = quality_summary['passed_checks']
    critical_failures = quality_summary['severity_counts']['critical']
    error_failures = quality_summary['severity_counts']['error']

    if critical_failures > 0:
        quality_score = 0.0  # 치명적 오류가 있으면 0점
    elif error_failures > 0:
        quality_score = max(0.0, 40.0 - error_failures * 10)  # 에러당 10점 감점
    else:
        quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100.0

    validation_passed = critical_failures == 0 and error_failures == 0

    # S3에 처리된 데이터 저장
    processed_data_key = None
    quality_report_key = None

    if s3_manager:
        try:
            # 처리된 데이터 저장
            processed_data_key = s3_manager.store_processed_features(
                df=df,
                feature_set_name="weather_observations",
                version="v1",
                timestamp=ingest_result["end_time"]
            )

            # 품질 리포트 저장
            import json
            quality_report_key = s3_manager._build_key(
                "monitoring",
                "quality",
                ingest_result["station_id"],
                ingest_result["end_time"].strftime("%Y/%m/%d"),
                f"quality_{ingest_result['end_time'].strftime('%H%M')}.json"
            )

            s3_manager.s3_client.put_object(
                Bucket=s3_manager.bucket_name,
                Key=quality_report_key,
                Body=json.dumps(quality_report, indent=2, default=str).encode("utf-8"),
                Metadata={
                    "station_id": ingest_result["station_id"],
                    "timestamp": ingest_result["end_time"].isoformat(),
                    "quality_score": str(quality_score),
                    "validation_passed": str(validation_passed),
                },
                ContentType="application/json",
            )

            logger.info(f"Stored processed data to S3: {processed_data_key}")
            logger.info(f"Stored quality report to S3: {quality_report_key}")

        except Exception as e:
            logger.warning(f"Failed to store processed data to S3: {e}")

    # 로컬에도 저장
    end_time = ingest_result["end_time"]
    parquet_path = storage.interim_path(
        "weather_normalized",
        "kma",
        ingest_result["station_id"],
        end_time.strftime("%Y"),
        end_time.strftime("%m"),
        end_time.strftime("%d"),
        f"{end_time.strftime('%H%M')}.parquet",
    )
    storage.write_dataframe(parquet_path, df)

    quality_json_path = storage.interim_path(
        "quality_reports",
        "kma",
        ingest_result["station_id"],
        end_time.strftime("%Y"),
        end_time.strftime("%m"),
        end_time.strftime("%d"),
        f"{end_time.strftime('%H%M')}_quality.json",
    )
    storage.write_json(quality_json_path, quality_report)

    return {
        "dataframe": df,
        "processed_data_key": processed_data_key,
        "quality_report_key": quality_report_key,
        "local_parquet_path": parquet_path,
        "local_quality_path": quality_json_path,
        "quality_summary": quality_summary,
        "quality_score": quality_score,
        "validation_passed": validation_passed,
        "validation_results": validation_results
    }


@task
def create_features_with_validation(
    process_result: Dict[str, Any],
    ingest_result: Dict[str, Any],
    storage: StorageManager,
    s3_manager: Optional[S3StorageManager] = None,
    *,
    target_period: str = "morning_commute",
    lookback_hours: int = 3,
    version: str = "v1"
) -> Dict[str, Any]:
    """검증된 데이터로부터 피처 생성"""
    logger = get_run_logger()

    # 검증 실패 시 처리
    if not process_result["validation_passed"]:
        logger.warning("Data validation failed, creating features with warnings")

    df = process_result["dataframe"]
    end_time = ingest_result["end_time"]

    # 피처 엔지니어링 (기존 로직 사용)
    from ..data_sources.weather_api import WeatherObservation
    from ..pipelines.baseline import compute_commute_comfort_score

    tz = df["timestamp"].iloc[0].tzinfo or ZoneInfo("Asia/Seoul")
    window_end = end_time.astimezone(tz)
    window_start = window_end - dt.timedelta(hours=lookback_hours - 1)

    mask = (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
    window_df = df.loc[mask]

    if window_df.empty:
        raise ValueError("No observations available in requested window for feature engineering")

    temp_series = window_df["temperature_c"].astype(float)
    humidity_series = window_df["relative_humidity"].astype(float)

    features = {
        "event_time": [window_end],
        "target_period": [target_period],
        "temp_median_3h": [float(temp_series.median())],
        "temp_range_3h": [float(temp_series.max() - temp_series.min())],
        "wind_peak_3h": [float(window_df["wind_speed_ms"].astype(float).max())],
        "precip_total_3h": [float(window_df["precipitation_mm"].astype(float).sum())],
        "precip_type_dominant": [window_df["precipitation_type"].mode().iat[0]],
        "humidity_avg_3h": [float(humidity_series.mean()) if not humidity_series.isna().all() else None],
        "data_version": [version],
        "quality_score": [process_result["quality_score"]],
        "validation_passed": [process_result["validation_passed"]],
    }

    # 베이스라인 컴포트 스코어 계산
    observations = []
    for _, row in window_df.iterrows():
        humidity_value = row["relative_humidity"]
        if pd.isna(humidity_value):
            humidity = None
        else:
            humidity = float(humidity_value)

        observations.append(
            WeatherObservation(
                timestamp=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                temperature_c=float(row["temperature_c"]),
                wind_speed_ms=float(row["wind_speed_ms"]),
                precipitation_mm=float(row["precipitation_mm"]),
                relative_humidity=humidity,
                precipitation_type=str(row["precipitation_type"]),
            )
        )

    comfort = compute_commute_comfort_score(observations)
    features["baseline_comfort_score"] = [comfort.score]
    features["label_bucket"] = [comfort.label]

    features_df = pd.DataFrame(features)

    logger.info(f"Created features with quality score: {process_result['quality_score']:.1f}")

    # S3에 피처 저장
    feature_data_key = None
    if s3_manager:
        try:
            feature_data_key = s3_manager.store_processed_features(
                df=features_df,
                feature_set_name="commute_features",
                version=version,
                timestamp=window_end
            )
            logger.info(f"Stored features to S3: {feature_data_key}")
        except Exception as e:
            logger.warning(f"Failed to store features to S3: {e}")

    # 로컬 저장
    feature_path = storage.processed_path(
        "features",
        "commute",
        version,
        window_end.strftime("%Y"),
        window_end.strftime("%m"),
        window_end.strftime("%d"),
        f"{window_end.strftime('%H%M')}.parquet",
    )
    storage.write_dataframe(feature_path, features_df)

    return {
        "features_df": features_df,
        "feature_data_key": feature_data_key,
        "local_feature_path": feature_path,
        "comfort_score": comfort.score,
        "comfort_label": comfort.label
    }


@flow(name="enhanced-weather-data-pipeline")
def enhanced_weather_data_flow(
    *,
    project_root: Optional[Path] = None,
    auth_key: Optional[str] = None,
    station_id: str = "108",
    base_url: str = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
    target_period: str = "morning_commute",
    lookback_hours: int = 3,
    data_version: str = "v1",
    run_time: Optional[dt.datetime] = None,
    use_s3: bool = True,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
) -> EnhancedDataArtifact:
    """강화된 기상 데이터 파이프라인"""

    logger = get_run_logger()
    paths = ProjectPaths.from_root(project_root or Path(__file__).resolve().parents[3])

    resolved_run_time = run_time or dt.datetime.now(ZoneInfo("Asia/Seoul"))
    logger.info("Starting enhanced weather data pipeline", extra={"run_time": resolved_run_time.isoformat()})

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
            use_s3 = False

    # 파이프라인 실행
    ingest_future = ingest_weather_data_with_quality(
        kma_config, storage, s3_manager,
        run_time=resolved_run_time,
        lookback_hours=lookback_hours
    )

    process_future = process_and_validate_data(
        ingest_future, storage, s3_manager,
        timezone=kma_config.timezone
    )

    feature_future = create_features_with_validation(
        process_future, ingest_future, storage, s3_manager,
        target_period=target_period,
        lookback_hours=lookback_hours,
        version=data_version
    )

    # 결과 수집
    ingest_result = ingest_future.result()
    process_result = process_future.result()
    feature_result = feature_future.result()

    # 아티팩트 생성
    artifact = EnhancedDataArtifact(
        raw_data_key=ingest_result.get("raw_data_key", ""),
        processed_data_key=process_result.get("processed_data_key", ""),
        quality_report_key=process_result.get("quality_report_key", ""),
        feature_data_key=feature_result.get("feature_data_key", ""),
        timestamp=resolved_run_time,
        station_id=station_id,
        data_quality_score=process_result["quality_score"],
        validation_passed=process_result["validation_passed"],
        row_count=len(process_result["dataframe"])
    )

    logger.info(
        f"Pipeline completed successfully. Quality score: {artifact.data_quality_score:.1f}, "
        f"Validation passed: {artifact.validation_passed}"
    )

    return artifact


if __name__ == "__main__":
    # 테스트 실행
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        result = enhanced_weather_data_flow()
        print(f"Pipeline result: {result}")
    else:
        print("Use 'python enhanced_data_pipeline.py test' to run a test")