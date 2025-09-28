"""실시간 ASOS 기상 데이터 수집기."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from ..config import KMAAPIConfig
from ..data_quality import DataQualityValidator, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class AsosObservation:
    """ASOS 기상 관측 데이터"""
    timestamp: datetime
    station_id: str

    # 핵심 기상 요소
    temperature: Optional[float]  # 기온 (°C)
    humidity: Optional[float]     # 상대습도 (%)
    pressure: Optional[float]     # 해면기압 (hPa)
    wind_speed: Optional[float]   # 풍속 (m/s)
    wind_direction: Optional[float]  # 풍향 (도)
    precipitation_1h: Optional[float]  # 1시간 강수량 (mm)
    precipitation_type: Optional[str]  # 강수형태
    visibility: Optional[float]   # 시정 (km)
    cloud_cover: Optional[float]  # 운량 (0-10)

    # 추가 기상 요소
    dew_point: Optional[float]    # 이슬점온도 (°C)
    wind_gust: Optional[float]    # 최대풍속 (m/s)
    solar_radiation: Optional[float]  # 일사량 (MJ/m²)
    ground_temp: Optional[float]  # 지면온도 (°C)

    # 메타데이터
    data_quality: Optional[float] = None
    source: str = "asos"


class AsosDataCollector:
    """ASOS 데이터 수집기"""

    def __init__(self, kma_config: KMAAPIConfig):
        self.kma_config = kma_config
        self.session = requests.Session()
        self.validator = DataQualityValidator()

        # 핵심 피처 목록 (중요도 순)
        self.core_features = [
            "temperature",
            "humidity",
            "precipitation_1h",
            "wind_speed",
            "pressure",
            "visibility",
            "wind_direction",
            "cloud_cover"
        ]

        # 부가 피처 목록
        self.additional_features = [
            "dew_point",
            "wind_gust",
            "solar_radiation",
            "ground_temp"
        ]

    def fetch_latest_observation(
        self,
        station_id: Optional[str] = None,
        timeout: int = 30
    ) -> Optional[AsosObservation]:
        """최신 기상 관측 데이터 수집"""

        target_station = station_id or self.kma_config.station_id

        # 현재 시간 기준으로 최근 완료된 정시 시간 계산
        now = datetime.now(ZoneInfo(self.kma_config.timezone))
        target_time = now.replace(minute=0, second=0, microsecond=0)

        # 아직 해당 시간 데이터가 준비되지 않았을 가능성을 고려하여 1시간 전 데이터 요청
        if now.minute < 10:  # 정시 후 10분 이내라면 이전 시간 데이터 요청
            target_time -= timedelta(hours=1)

        logger.info(f"Fetching ASOS data for station {target_station} at {target_time}")

        try:
            # KMA API 호출
            params = {
                "tm1": target_time.strftime("%Y%m%d%H%M"),
                "tm2": target_time.strftime("%Y%m%d%H%M"),
                "stn": target_station,
                "help": self.kma_config.help_flag,
                "authKey": self.kma_config.auth_key,
            }

            response = self.session.get(
                self.kma_config.base_url,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()

            # 응답 파싱
            observation = self._parse_asos_response(response.text, target_station, target_time)

            if observation:
                logger.info(f"Successfully collected ASOS data for {target_time}")
                return observation
            else:
                logger.warning(f"No valid data found for {target_time}")
                return None

        except requests.RequestException as e:
            logger.error(f"Failed to fetch ASOS data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing ASOS data: {e}")
            return None

    def _parse_asos_response(
        self,
        response_text: str,
        station_id: str,
        target_time: datetime
    ) -> Optional[AsosObservation]:
        """ASOS 응답 파싱"""

        try:
            # 기존 KMA 파싱 로직 활용하되 ASOS 형식에 맞게 확장
            from .kma_api import normalize_kma_observations

            # 기본 파싱
            observations = normalize_kma_observations(response_text)

            if not observations:
                return None

            # 가장 최근 관측 데이터 선택
            latest_obs = observations[-1]

            # AsosObservation 객체 생성
            asos_obs = AsosObservation(
                timestamp=latest_obs.timestamp,
                station_id=station_id,
                temperature=latest_obs.temperature_c,
                humidity=latest_obs.relative_humidity,
                pressure=None,  # KMA API에서 제공되지 않음
                wind_speed=latest_obs.wind_speed_ms,
                wind_direction=None,  # KMA API에서 제공되지 않음
                precipitation_1h=latest_obs.precipitation_mm,
                precipitation_type=latest_obs.precipitation_type,
                visibility=None,  # KMA API에서 제공되지 않음
                cloud_cover=None,  # KMA API에서 제공되지 않음
                dew_point=None,
                wind_gust=None,
                solar_radiation=None,
                ground_temp=None,
            )

            # 데이터 품질 평가
            asos_obs.data_quality = self._assess_data_quality(asos_obs)

            return asos_obs

        except Exception as e:
            logger.error(f"Failed to parse ASOS response: {e}")
            return None

    def _assess_data_quality(self, observation: AsosObservation) -> float:
        """데이터 품질 평가 (0-100 점수)"""

        quality_score = 100.0

        # 핵심 피처 완성도 확인
        core_completeness = 0
        for feature in self.core_features:
            value = getattr(observation, feature, None)
            if value is not None and not pd.isna(value):
                core_completeness += 1

        core_ratio = core_completeness / len(self.core_features)

        # 완성도에 따른 점수 조정
        if core_ratio < 0.5:
            quality_score *= 0.3  # 핵심 피처 50% 미만이면 심각한 품질 문제
        elif core_ratio < 0.7:
            quality_score *= 0.6  # 70% 미만이면 품질 저하
        elif core_ratio < 0.9:
            quality_score *= 0.8  # 90% 미만이면 약간의 품질 저하

        # 값 범위 검증
        if observation.temperature is not None:
            if observation.temperature < -50 or observation.temperature > 60:
                quality_score *= 0.8

        if observation.humidity is not None:
            if observation.humidity < 0 or observation.humidity > 100:
                quality_score *= 0.8

        if observation.wind_speed is not None:
            if observation.wind_speed < 0 or observation.wind_speed > 100:
                quality_score *= 0.8

        # 시간 신선도 확인
        if observation.timestamp:
            time_diff = datetime.now(ZoneInfo("Asia/Seoul")) - observation.timestamp
            if time_diff > timedelta(hours=2):
                quality_score *= 0.9  # 2시간 이상 된 데이터는 약간 점수 감점

        return max(0.0, min(100.0, quality_score))

    def select_core_features(
        self,
        observation: AsosObservation,
        min_quality_threshold: float = 70.0
    ) -> Dict[str, Any]:
        """핵심 피처 선택 및 추출"""

        if observation.data_quality < min_quality_threshold:
            logger.warning(
                f"Data quality ({observation.data_quality:.1f}) below threshold "
                f"({min_quality_threshold})"
            )

        # 핵심 피처 추출
        core_data = {}

        # 기본 메타데이터
        core_data["timestamp"] = observation.timestamp
        core_data["station_id"] = observation.station_id
        core_data["data_quality"] = observation.data_quality
        core_data["source"] = observation.source

        # 핵심 기상 피처 (중요도 순으로 선택)
        feature_mapping = {
            "temperature": "temperature_c",
            "humidity": "relative_humidity_pct",
            "precipitation_1h": "precipitation_mm_1h",
            "wind_speed": "wind_speed_ms",
            "pressure": "pressure_hpa",
            "visibility": "visibility_km",
            "wind_direction": "wind_direction_deg",
            "cloud_cover": "cloud_cover_tenths"
        }

        for original_name, standard_name in feature_mapping.items():
            value = getattr(observation, original_name, None)
            if value is not None and not pd.isna(value):
                core_data[standard_name] = float(value)
            else:
                core_data[standard_name] = None

        # 피처 중요도 스코어링
        core_data["feature_completeness"] = self._calculate_feature_completeness(core_data)
        core_data["feature_reliability"] = self._calculate_feature_reliability(observation)

        return core_data

    def _calculate_feature_completeness(self, core_data: Dict[str, Any]) -> float:
        """피처 완성도 계산"""

        feature_columns = [
            "temperature_c", "relative_humidity_pct", "precipitation_mm_1h",
            "wind_speed_ms", "pressure_hpa", "visibility_km"
        ]

        available_count = sum(
            1 for col in feature_columns
            if col in core_data and core_data[col] is not None
        )

        return available_count / len(feature_columns) * 100.0

    def _calculate_feature_reliability(self, observation: AsosObservation) -> float:
        """피처 신뢰성 계산"""

        reliability_score = 100.0

        # 값의 일관성 검사
        if (observation.temperature is not None and
            observation.humidity is not None):
            # 온도와 습도의 관계 검증
            if observation.temperature > 35 and observation.humidity > 90:
                reliability_score *= 0.9  # 극한 조건의 조합은 신뢰성 감소

        if (observation.precipitation_1h is not None and
            observation.humidity is not None):
            # 강수와 습도의 관계 검증
            if observation.precipitation_1h > 0 and observation.humidity < 50:
                reliability_score *= 0.8  # 비가 와도 습도가 낮으면 신뢰성 의심

        # 데이터 변화율 검사 (이전 데이터와 비교, 별도 구현 필요)
        # 여기서는 기본 신뢰성만 계산

        return max(0.0, min(100.0, reliability_score))

    def collect_hourly_batch(
        self,
        hours_back: int = 24,
        station_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """지난 N시간의 데이터 일괄 수집"""

        target_station = station_id or self.kma_config.station_id
        collected_data = []

        now = datetime.now(ZoneInfo(self.kma_config.timezone))

        for i in range(hours_back):
            target_time = now - timedelta(hours=i+1)
            target_time = target_time.replace(minute=0, second=0, microsecond=0)

            try:
                # 시간당 0.5초 대기 (API 부하 방지)
                if i > 0:
                    time.sleep(0.5)

                observation = self.fetch_historical_observation(target_station, target_time)

                if observation and observation.data_quality > 50.0:
                    core_data = self.select_core_features(observation)
                    collected_data.append(core_data)
                    logger.info(f"Collected data for {target_time}: quality={observation.data_quality:.1f}")
                else:
                    logger.warning(f"Poor quality data for {target_time}, skipping")

            except Exception as e:
                logger.error(f"Failed to collect data for {target_time}: {e}")
                continue

        logger.info(f"Collected {len(collected_data)} valid observations out of {hours_back} attempts")
        return collected_data

    def fetch_historical_observation(
        self,
        station_id: str,
        target_time: datetime
    ) -> Optional[AsosObservation]:
        """특정 시간의 과거 관측 데이터 수집"""

        try:
            params = {
                "tm1": target_time.strftime("%Y%m%d%H%M"),
                "tm2": target_time.strftime("%Y%m%d%H%M"),
                "stn": station_id,
                "help": self.kma_config.help_flag,
                "authKey": self.kma_config.auth_key,
            }

            response = self.session.get(
                self.kma_config.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()

            return self._parse_asos_response(response.text, station_id, target_time)

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {target_time}: {e}")
            return None

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """피처 중요도 순위 반환"""

        # 도메인 지식 기반 피처 중요도
        importance_scores = {
            "temperature_c": 1.0,           # 가장 중요한 기상 요소
            "precipitation_mm_1h": 0.95,   # 출퇴근에 직접적 영향
            "relative_humidity_pct": 0.85,  # 체감 온도에 영향
            "wind_speed_ms": 0.80,         # 체감 온도와 쾌적성에 영향
            "visibility_km": 0.70,         # 교통 안전에 영향
            "pressure_hpa": 0.60,          # 날씨 변화 예측에 유용
            "wind_direction_deg": 0.50,    # 부가적 정보
            "cloud_cover_tenths": 0.45,    # 간접적 영향
        }

        return sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    def create_feature_summary(self, core_data: Dict[str, Any]) -> Dict[str, Any]:
        """피처 요약 정보 생성"""

        summary = {
            "timestamp": core_data.get("timestamp"),
            "data_quality": core_data.get("data_quality", 0.0),
            "feature_count": sum(1 for k, v in core_data.items()
                               if k.endswith(('_c', '_pct', '_mm_1h', '_ms', '_hpa', '_km', '_deg', '_tenths'))
                               and v is not None),
            "completeness": core_data.get("feature_completeness", 0.0),
            "reliability": core_data.get("feature_reliability", 0.0),
        }

        # 주요 기상 상황 요약
        temp = core_data.get("temperature_c")
        precip = core_data.get("precipitation_mm_1h")
        wind = core_data.get("wind_speed_ms")

        conditions = []
        if temp is not None:
            if temp < 0:
                conditions.append("freezing")
            elif temp > 30:
                conditions.append("hot")

        if precip is not None and precip > 0:
            if precip > 10:
                conditions.append("heavy_rain")
            else:
                conditions.append("light_rain")

        if wind is not None and wind > 10:
            conditions.append("windy")

        summary["weather_conditions"] = conditions
        summary["severity_level"] = len(conditions)  # 기상 심각도

        return summary