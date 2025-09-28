"""날씨 데이터 피처 엔지니어링."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseFeatureTransformer, FeatureMetadata

logger = logging.getLogger(__name__)


class BasicWeatherFeatures(BaseFeatureTransformer):
    """기본 날씨 피처 생성"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("basic_weather_features", version)

        self.feature_metadata = [
            FeatureMetadata(
                name="temp_median_3h",
                description="3시간 온도 중앙값",
                feature_type="numerical",
                source_columns=["temperature_c"],
                creation_logic="pd.median() over 3-hour window",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(-50.0, 60.0),
                nullable=False
            ),
            FeatureMetadata(
                name="temp_range_3h",
                description="3시간 온도 범위 (최고-최저)",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="max() - min() over 3-hour window",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 40.0),
                nullable=False
            ),
            FeatureMetadata(
                name="wind_peak_3h",
                description="3시간 최대 풍속",
                feature_type="numerical",
                source_columns=["wind_speed_ms"],
                creation_logic="max() over 3-hour window",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=False
            ),
            FeatureMetadata(
                name="precip_total_3h",
                description="3시간 총 강수량",
                feature_type="numerical",
                source_columns=["precipitation_mm"],
                creation_logic="sum() over 3-hour window",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 1000.0),
                nullable=False
            ),
            FeatureMetadata(
                name="humidity_avg_3h",
                description="3시간 평균 습도",
                feature_type="numerical",
                source_columns=["relative_humidity"],
                creation_logic="mean() over 3-hour window",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=True
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 집계 피처 생성"""
        result_df = df.copy()

        # 온도 피처
        if "temperature_c" in df.columns:
            result_df["temp_median_3h"] = df["temperature_c"].median()
            result_df["temp_range_3h"] = df["temperature_c"].max() - df["temperature_c"].min()

        # 풍속 피처
        if "wind_speed_ms" in df.columns:
            result_df["wind_peak_3h"] = df["wind_speed_ms"].max()

        # 강수량 피처
        if "precipitation_mm" in df.columns:
            result_df["precip_total_3h"] = df["precipitation_mm"].sum()

        # 습도 피처
        if "relative_humidity" in df.columns:
            result_df["humidity_avg_3h"] = df["relative_humidity"].mean()

        return result_df

    def get_feature_names(self) -> List[str]:
        return ["temp_median_3h", "temp_range_3h", "wind_peak_3h", "precip_total_3h", "humidity_avg_3h"]


class AdvancedWeatherFeatures(BaseFeatureTransformer):
    """고급 날씨 피처 생성"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("advanced_weather_features", version)

        self.feature_metadata = [
            FeatureMetadata(
                name="temp_stability",
                description="온도 안정성 (표준편차 역수)",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="1 / (std() + 1e-6)",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=False
            ),
            FeatureMetadata(
                name="wind_variability",
                description="풍속 변동성 (변동계수)",
                feature_type="derived",
                source_columns=["wind_speed_ms"],
                creation_logic="std() / (mean() + 1e-6)",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 10.0),
                nullable=False
            ),
            FeatureMetadata(
                name="precip_intensity",
                description="강수 강도 (0: 없음, 1: 약함, 2: 보통, 3: 강함)",
                feature_type="categorical",
                source_columns=["precipitation_mm"],
                creation_logic="Binning: 0, 0-1, 1-5, 5+ mm/h",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 3),
                nullable=False
            ),
            FeatureMetadata(
                name="comfort_temp_score",
                description="온도 쾌적도 점수 (0-100)",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="Gaussian scoring around 20°C",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=False
            ),
            FeatureMetadata(
                name="weather_severity",
                description="날씨 심각도 종합 점수",
                feature_type="derived",
                source_columns=["temperature_c", "wind_speed_ms", "precipitation_mm"],
                creation_logic="Weighted combination of extreme weather factors",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=False
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 피처 생성"""
        result_df = df.copy()

        # 온도 안정성
        if "temperature_c" in df.columns:
            temp_std = df["temperature_c"].std()
            result_df["temp_stability"] = 1.0 / (temp_std + 1e-6)

            # 온도 쾌적도 점수 (20°C 중심의 가우시안)
            temp_median = df["temperature_c"].median()
            comfort_score = 100 * np.exp(-((temp_median - 20) ** 2) / (2 * 10 ** 2))
            result_df["comfort_temp_score"] = max(0, min(100, comfort_score))

        # 풍속 변동성
        if "wind_speed_ms" in df.columns:
            wind_mean = df["wind_speed_ms"].mean()
            wind_std = df["wind_speed_ms"].std()
            result_df["wind_variability"] = wind_std / (wind_mean + 1e-6)

        # 강수 강도 분류
        if "precipitation_mm" in df.columns:
            precip_max = df["precipitation_mm"].max()
            if precip_max == 0:
                intensity = 0
            elif precip_max <= 1:
                intensity = 1
            elif precip_max <= 5:
                intensity = 2
            else:
                intensity = 3
            result_df["precip_intensity"] = intensity

        # 날씨 심각도 종합 점수
        severity_score = 0

        if "temperature_c" in df.columns:
            temp_median = df["temperature_c"].median()
            # 극한 온도 페널티
            if temp_median < 0 or temp_median > 35:
                severity_score += 30
            elif temp_median < 5 or temp_median > 30:
                severity_score += 15

        if "wind_speed_ms" in df.columns:
            wind_max = df["wind_speed_ms"].max()
            # 강풍 페널티
            if wind_max > 15:
                severity_score += 25
            elif wind_max > 10:
                severity_score += 10

        if "precipitation_mm" in df.columns:
            precip_total = df["precipitation_mm"].sum()
            # 강수 페널티
            if precip_total > 10:
                severity_score += 20
            elif precip_total > 5:
                severity_score += 10

        result_df["weather_severity"] = min(100, severity_score)

        return result_df

    def get_feature_names(self) -> List[str]:
        return [
            "temp_stability",
            "wind_variability",
            "precip_intensity",
            "comfort_temp_score",
            "weather_severity"
        ]


class TemporalWeatherFeatures(BaseFeatureTransformer):
    """시계열 날씨 피처 생성"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("temporal_weather_features", version)

        self.feature_metadata = [
            FeatureMetadata(
                name="hour_of_day",
                description="시간 (0-23)",
                feature_type="temporal",
                source_columns=["timestamp"],
                creation_logic="Extract hour from timestamp",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 23),
                nullable=False
            ),
            FeatureMetadata(
                name="day_of_week",
                description="요일 (0=월요일, 6=일요일)",
                feature_type="temporal",
                source_columns=["timestamp"],
                creation_logic="Extract weekday from timestamp",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 6),
                nullable=False
            ),
            FeatureMetadata(
                name="month_of_year",
                description="월 (1-12)",
                feature_type="temporal",
                source_columns=["timestamp"],
                creation_logic="Extract month from timestamp",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(1, 12),
                nullable=False
            ),
            FeatureMetadata(
                name="is_weekend",
                description="주말 여부 (0: 평일, 1: 주말)",
                feature_type="categorical",
                source_columns=["timestamp"],
                creation_logic="weekday >= 5",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 1),
                nullable=False
            ),
            FeatureMetadata(
                name="is_rush_hour",
                description="출퇴근 시간 여부 (7-9시, 18-20시)",
                feature_type="categorical",
                source_columns=["timestamp"],
                creation_logic="hour in [7,8,9,18,19,20]",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 1),
                nullable=False
            ),
            FeatureMetadata(
                name="season",
                description="계절 (0: 봄, 1: 여름, 2: 가을, 3: 겨울)",
                feature_type="categorical",
                source_columns=["timestamp"],
                creation_logic="Month-based season mapping",
                version=version,
                created_at=datetime.now(),
                data_type="int",
                expected_range=(0, 3),
                nullable=False
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 피처 생성"""
        result_df = df.copy()

        if "timestamp" not in df.columns:
            # event_time을 timestamp로 사용
            if "event_time" in df.columns:
                timestamp_col = "event_time"
            else:
                logger.warning("No timestamp column found, using current time")
                result_df["timestamp"] = datetime.now()
                timestamp_col = "timestamp"
        else:
            timestamp_col = "timestamp"

        # 타임스탬프를 datetime으로 변환
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # 대표 시간 선택 (첫 번째 행의 시간 사용)
        if len(df) > 0:
            ref_time = result_df[timestamp_col].iloc[0]
        else:
            ref_time = datetime.now()

        # 시간 관련 피처
        result_df["hour_of_day"] = ref_time.hour
        result_df["day_of_week"] = ref_time.weekday()
        result_df["month_of_year"] = ref_time.month

        # 주말 여부
        result_df["is_weekend"] = 1 if ref_time.weekday() >= 5 else 0

        # 출퇴근 시간 여부
        rush_hours = [7, 8, 9, 18, 19, 20]
        result_df["is_rush_hour"] = 1 if ref_time.hour in rush_hours else 0

        # 계절 (3-5월: 봄, 6-8월: 여름, 9-11월: 가을, 12-2월: 겨울)
        month = ref_time.month
        if 3 <= month <= 5:
            season = 0  # 봄
        elif 6 <= month <= 8:
            season = 1  # 여름
        elif 9 <= month <= 11:
            season = 2  # 가을
        else:
            season = 3  # 겨울

        result_df["season"] = season

        return result_df

    def get_feature_names(self) -> List[str]:
        return [
            "hour_of_day",
            "day_of_week",
            "month_of_year",
            "is_weekend",
            "is_rush_hour",
            "season"
        ]


class ComfortScoreFeatures(BaseFeatureTransformer):
    """쾌적도 관련 피처 생성"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("comfort_score_features", version)

        self.feature_metadata = [
            FeatureMetadata(
                name="temp_comfort_penalty",
                description="온도 불쾌감 페널티 (0-40점)",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="Distance-based penalty from optimal range (10-25°C)",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 40.0),
                nullable=False
            ),
            FeatureMetadata(
                name="precip_comfort_penalty",
                description="강수 불쾌감 페널티 (0-35점)",
                feature_type="derived",
                source_columns=["precipitation_mm"],
                creation_logic="Precipitation-based penalty with intensity scaling",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 35.0),
                nullable=False
            ),
            FeatureMetadata(
                name="wind_comfort_penalty",
                description="풍속 불쾌감 페널티 (0-25점)",
                feature_type="derived",
                source_columns=["wind_speed_ms"],
                creation_logic="Wind speed penalty above 4 m/s",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 25.0),
                nullable=False
            ),
            FeatureMetadata(
                name="humidity_comfort_penalty",
                description="습도 불쾌감 페널티 (0-15점)",
                feature_type="derived",
                source_columns=["relative_humidity"],
                creation_logic="Humidity penalty outside 30-70% range",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 15.0),
                nullable=True
            ),
            FeatureMetadata(
                name="predicted_comfort_score",
                description="예측된 쾌적도 점수 (0-100)",
                feature_type="derived",
                source_columns=["temperature_c", "precipitation_mm", "wind_speed_ms", "relative_humidity"],
                creation_logic="100 - sum of all comfort penalties",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 100.0),
                nullable=False
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """쾌적도 피처 생성"""
        result_df = df.copy()

        # 온도 페널티
        temp_penalty = 0
        if "temperature_c" in df.columns:
            temp_median = df["temperature_c"].median()
            if temp_median < 10:
                temp_penalty = min(40, (10 - temp_median) * 2)
            elif temp_median > 25:
                temp_penalty = min(40, (temp_median - 25) * 1.5)

        result_df["temp_comfort_penalty"] = temp_penalty

        # 강수 페널티
        precip_penalty = 0
        if "precipitation_mm" in df.columns:
            precip_total = df["precipitation_mm"].sum()
            if precip_total > 0:
                if precip_total <= 1:
                    precip_penalty = 10
                elif precip_total <= 5:
                    precip_penalty = 20
                else:
                    precip_penalty = 35

        result_df["precip_comfort_penalty"] = precip_penalty

        # 풍속 페널티
        wind_penalty = 0
        if "wind_speed_ms" in df.columns:
            wind_max = df["wind_speed_ms"].max()
            if wind_max > 4:
                wind_penalty = min(25, (wind_max - 4) * 3)

        result_df["wind_comfort_penalty"] = wind_penalty

        # 습도 페널티
        humidity_penalty = 0
        if "relative_humidity" in df.columns and not df["relative_humidity"].isna().all():
            humidity_avg = df["relative_humidity"].mean()
            if not pd.isna(humidity_avg):
                if humidity_avg < 30:
                    humidity_penalty = min(15, (30 - humidity_avg) * 0.3)
                elif humidity_avg > 70:
                    humidity_penalty = min(15, (humidity_avg - 70) * 0.5)

        result_df["humidity_comfort_penalty"] = humidity_penalty

        # 전체 쾌적도 점수
        total_penalty = temp_penalty + precip_penalty + wind_penalty + humidity_penalty
        comfort_score = max(0, 100 - total_penalty)
        result_df["predicted_comfort_score"] = comfort_score

        return result_df

    def get_feature_names(self) -> List[str]:
        return [
            "temp_comfort_penalty",
            "precip_comfort_penalty",
            "wind_comfort_penalty",
            "humidity_comfort_penalty",
            "predicted_comfort_score"
        ]


class StatisticalFeatures(BaseFeatureTransformer):
    """통계적 피처 생성"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("statistical_features", version)

        self.feature_metadata = [
            FeatureMetadata(
                name="temp_skewness",
                description="온도 분포 왜도",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="scipy.stats.skew()",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(-3.0, 3.0),
                nullable=False
            ),
            FeatureMetadata(
                name="wind_kurtosis",
                description="풍속 분포 첨도",
                feature_type="derived",
                source_columns=["wind_speed_ms"],
                creation_logic="scipy.stats.kurtosis()",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(-3.0, 10.0),
                nullable=False
            ),
            FeatureMetadata(
                name="data_completeness",
                description="데이터 완성도 (0-1)",
                feature_type="derived",
                source_columns=["temperature_c", "wind_speed_ms", "precipitation_mm", "relative_humidity"],
                creation_logic="Ratio of non-null values",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 1.0),
                nullable=False
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """통계적 피처 생성"""
        result_df = df.copy()

        # 온도 왜도
        if "temperature_c" in df.columns and len(df) > 2:
            temp_values = df["temperature_c"].dropna()
            if len(temp_values) > 2:
                result_df["temp_skewness"] = stats.skew(temp_values)
            else:
                result_df["temp_skewness"] = 0.0
        else:
            result_df["temp_skewness"] = 0.0

        # 풍속 첨도
        if "wind_speed_ms" in df.columns and len(df) > 2:
            wind_values = df["wind_speed_ms"].dropna()
            if len(wind_values) > 2:
                result_df["wind_kurtosis"] = stats.kurtosis(wind_values)
            else:
                result_df["wind_kurtosis"] = 0.0
        else:
            result_df["wind_kurtosis"] = 0.0

        # 데이터 완성도
        key_columns = ["temperature_c", "wind_speed_ms", "precipitation_mm", "relative_humidity"]
        available_columns = [col for col in key_columns if col in df.columns]

        if available_columns:
            total_values = len(df) * len(available_columns)
            non_null_values = df[available_columns].count().sum()
            completeness = non_null_values / total_values if total_values > 0 else 0.0
        else:
            completeness = 0.0

        result_df["data_completeness"] = completeness

        return result_df

    def get_feature_names(self) -> List[str]:
        return ["temp_skewness", "wind_kurtosis", "data_completeness"]