"""Prefect flow scaffolding for the commute weather MLOps pipeline."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from prefect import flow, get_run_logger, task
from zoneinfo import ZoneInfo

from ...config import KMAAPIConfig, ProjectPaths
from ...data_sources.kma_api import normalize_kma_observations
from ...data_sources.weather_api import WeatherObservation
from ...pipelines.baseline import compute_commute_comfort_score
from ...storage import StorageManager


@dataclass(frozen=True)
class RawIngestArtifact:
    """Carries metadata about the raw payload persisted to disk."""

    raw_path: str
    metadata_path: str
    window_start: dt.datetime
    window_end: dt.datetime
    station_id: str
    fetched_at: dt.datetime
    response_bytes: int
    payload_text: str


def _resolve_project_paths(project_root: Optional[Path]) -> ProjectPaths:
    root = project_root or Path(__file__).resolve().parents[4]
    return ProjectPaths.from_root(root)


@task
def ingest_kma_payload(
    paths: ProjectPaths,
    storage: StorageManager,
    kma_config: KMAAPIConfig,
    *,
    run_time: Optional[dt.datetime] = None,
    lookback_hours: int = 3,
) -> RawIngestArtifact:
    import requests

    tz = ZoneInfo(kma_config.timezone)
    effective_time = (run_time or dt.datetime.now(tz)).astimezone(tz)
    end_time = effective_time.replace(minute=0, second=0, microsecond=0)
    if end_time > effective_time:
        end_time -= dt.timedelta(hours=1)

    span = max(1, lookback_hours)
    start_time = end_time - dt.timedelta(hours=span - 1)

    params = {
        "tm1": start_time.strftime("%Y%m%d%H%M"),
        "tm2": end_time.strftime("%Y%m%d%H%M"),
        "stn": kma_config.station_id,
        "help": kma_config.help_flag,
        "authKey": kma_config.auth_key,
    }

    session = requests.Session()
    response = session.get(kma_config.base_url, params=params, timeout=kma_config.timeout_seconds)
    response.raise_for_status()
    payload_text = response.text
    fetched_at = dt.datetime.now(dt.timezone.utc)

    filename = f"{end_time.strftime('%H%M')}"
    base_parts = (
        "kma",
        kma_config.station_id,
        start_time.strftime("%Y"),
        start_time.strftime("%m"),
        start_time.strftime("%d"),
    )
    raw_path = storage.raw_path(*base_parts, f"{filename}.txt")
    storage.write_text(raw_path, payload_text)

    metadata = {
        "fetched_at": fetched_at.isoformat(),
        "station_id": kma_config.station_id,
        "window_start": start_time.isoformat(),
        "window_end": end_time.isoformat(),
        "request_params": params,
        "response_bytes": len(payload_text.encode("utf-8")),
    }
    metadata_path = storage.raw_path(*base_parts, f"{filename}.meta.json")
    storage.write_json(metadata_path, metadata)

    return RawIngestArtifact(
        raw_path=raw_path,
        metadata_path=metadata_path,
        window_start=start_time,
        window_end=end_time,
        station_id=kma_config.station_id,
        fetched_at=fetched_at,
        response_bytes=len(payload_text.encode("utf-8")),
        payload_text=payload_text,
    )


def _normalize_timestamp(value: dt.datetime, timezone: str) -> dt.datetime:
    tz = ZoneInfo(timezone)
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


@task
def normalize_kma_payload(
    artifact: RawIngestArtifact,
    *,
    timezone: str,
    source: str = "kma",
) -> pd.DataFrame:
    observations = normalize_kma_observations(artifact.payload_text)

    rows = []
    for obs in observations:
        timestamp = _normalize_timestamp(obs.timestamp, timezone)
        rows.append(
            {
                "timestamp": timestamp,
                "station_id": artifact.station_id,
                "temperature_c": obs.temperature_c,
                "wind_speed_ms": obs.wind_speed_ms,
                "precipitation_mm": obs.precipitation_mm,
                "relative_humidity": obs.relative_humidity,
                "precipitation_type": obs.precipitation_type or "unknown",
                "source": source,
                "ingested_at": artifact.fetched_at,
            }
        )

    if not rows:
        raise ValueError("No observations parsed from KMA payload")

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    return df


@task
def validate_normalized_dataframe(df: pd.DataFrame) -> dict[str, float]:
    report: dict[str, float] = {
        "row_count": float(len(df)),
        "null_temperature": float(df["temperature_c"].isna().sum()),
        "null_wind_speed": float(df["wind_speed_ms"].isna().sum()),
        "null_precipitation": float(df["precipitation_mm"].isna().sum()),
    }
    if df["timestamp"].isna().any():
        raise ValueError("Timestamp column contains null values")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamp column must be sorted in ascending order")
    return report


@task
def persist_normalized_dataframe(
    df: pd.DataFrame,
    artifact: RawIngestArtifact,
    storage: StorageManager,
) -> str:
    parquet_path = storage.interim_path(
        "weather_normalized",
        "kma",
        artifact.station_id,
        artifact.window_start.strftime("%Y"),
        artifact.window_start.strftime("%m"),
        artifact.window_start.strftime("%d"),
        f"{artifact.window_end.strftime('%H%M')}.parquet",
    )
    return storage.write_dataframe(parquet_path, df)


def _to_weather_observations(df: pd.DataFrame) -> Sequence[WeatherObservation]:
    observations: list[WeatherObservation] = []
    for _, row in df.iterrows():
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
    return observations


@task
def build_commute_features(
    df: pd.DataFrame,
    *,
    run_time: dt.datetime,
    target_period: str,
    lookback_hours: int,
    version: str,
) -> pd.DataFrame:
    tz = df["timestamp"].iloc[0].tzinfo or ZoneInfo("Asia/Seoul")
    window_end = run_time.astimezone(tz)
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
    }

    observations = _to_weather_observations(window_df)
    comfort = compute_commute_comfort_score(observations)
    features["baseline_comfort_score"] = [comfort.score]
    features["label_bucket"] = [comfort.label]

    features_df = pd.DataFrame(features)
    return features_df


@task
def persist_features(
    df: pd.DataFrame,
    storage: StorageManager,
    *,
    version: str,
) -> str:
    event_time: dt.datetime = df["event_time"].iloc[0]
    feature_path = storage.processed_path(
        "features",
        "commute",
        version,
        event_time.strftime("%Y"),
        event_time.strftime("%m"),
        event_time.strftime("%d"),
        f"{event_time.strftime('%H%M')}.parquet",
    )
    return storage.write_dataframe(feature_path, df)


@task
def persist_validation_report(
    report: dict[str, float],
    artifact: RawIngestArtifact,
    storage: StorageManager,
) -> str:
    report_path = storage.interim_path(
        "quality_reports",
        "kma",
        artifact.station_id,
        artifact.window_start.strftime("%Y"),
        artifact.window_start.strftime("%m"),
        artifact.window_start.strftime("%d"),
        f"{artifact.window_end.strftime('%H%M')}.json",
    )
    storage.write_json(report_path, report)
    return report_path


@task
def train_baseline_model(
    paths: ProjectPaths,
    storage: StorageManager,
    *,
    version: str,
    tracking_uri: Optional[str],
    experiment_name: str,
) -> dict[str, object]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    feature_files = storage.list_feature_files(version)
    if not feature_files:
        return {"status": "skipped", "reason": "no_feature_files"}

    feature_frames = [storage.read_parquet(path) for path in feature_files]

    data = pd.concat(feature_frames, ignore_index=True).dropna(subset=["baseline_comfort_score"])
    if len(data) < 24:
        return {"status": "skipped", "reason": "insufficient_samples", "rows": len(data)}

    feature_columns = [
        "temp_median_3h",
        "temp_range_3h",
        "wind_peak_3h",
        "precip_total_3h",
        "humidity_avg_3h",
    ]
    usable = data.dropna(subset=feature_columns)
    if usable.empty:
        return {"status": "skipped", "reason": "insufficient_clean_samples"}

    X = usable[feature_columns]
    y = usable["baseline_comfort_score"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = mean_squared_error(y, predictions, squared=False)

    model_path = storage.models_path(
        "commute",
        f"baseline_regressor_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.pkl",
    )

    import mlflow
    import mlflow.sklearn

    resolved_tracking_uri = tracking_uri or f"file:{(paths.root / 'mlruns').as_posix()}"
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"baseline_{version}_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "version": version,
                "model": "GradientBoostingRegressor",
                "rows": len(X),
                "features": feature_columns,
            }
        )
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        storage.dump_joblib(model_path, {"model": model, "features": feature_columns, "version": version})

        return {
            "status": "trained",
            "model_path": str(model_path),
            "mlflow_run_id": run.info.run_id,
            "rmse": rmse,
            "training_rows": len(X),
            "tracking_uri": resolved_tracking_uri,
        }


@flow(name="commute-weather-ml-training")
def build_commute_training_flow(
    *,
    project_root: Optional[Path] = None,
    auth_key: Optional[str] = None,
    station_id: str = "108",
    base_url: str = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
    target_period: str = "morning_commute",
    lookback_hours: int = 3,
    data_version: str = "v1",
    run_time: Optional[dt.datetime] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "commute-baseline",
) -> dict[str, object]:
    """Orchestrate ingest → validate → feature → train steps using Prefect."""

    logger = get_run_logger()
    paths = _resolve_project_paths(project_root)

    resolved_run_time = run_time or dt.datetime.now(ZoneInfo("Asia/Seoul"))
    logger.info("Starting commute training flow", extra={"run_time": resolved_run_time.isoformat()})

    if auth_key is None:
        auth_key = os.getenv("KMA_AUTH_KEY")
    if auth_key is None:
        raise ValueError("KMA API auth key must be provided via parameter or KMA_AUTH_KEY environment variable")

    kma_config = KMAAPIConfig(
        base_url=base_url,
        auth_key=auth_key,
        station_id=station_id,
    )

    storage = StorageManager(
        paths,
        bucket=os.getenv("COMMUTE_S3_BUCKET"),
        prefix=os.getenv("COMMUTE_S3_PREFIX", ""),
    )

    raw_future = ingest_kma_payload(paths, storage, kma_config, run_time=resolved_run_time, lookback_hours=lookback_hours)
    normalized_future = normalize_kma_payload(raw_future, timezone=kma_config.timezone)
    validation_future = validate_normalized_dataframe(normalized_future)
    validation_path_future = persist_validation_report(validation_future, raw_future, storage)
    normalized_path_future = persist_normalized_dataframe(normalized_future, raw_future, storage)

    feature_future = build_commute_features(
        normalized_future,
        run_time=resolved_run_time,
        target_period=target_period,
        lookback_hours=lookback_hours,
        version=data_version,
    )
    feature_path_future = persist_features(feature_future, storage, version=data_version)

    training_future = train_baseline_model(
        paths,
        storage,
        version=data_version,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )

    raw_artifact = raw_future.result()
    normalized_path = normalized_path_future.result()
    validation_path = validation_path_future.result()
    feature_path = feature_path_future.result()
    training_result = training_future.result()

    return {
        "raw_path": str(raw_artifact.raw_path),
        "metadata_path": str(raw_artifact.metadata_path),
        "normalized_path": str(normalized_path),
        "validation_report": str(validation_path),
        "feature_path": str(feature_path),
        "training_result": training_result,
    }
