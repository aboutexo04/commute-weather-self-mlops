"""실제 KMA 데이터를 활용한 피처 엔지니어링 시스템."""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from zoneinfo import ZoneInfo

from ..data_sources.weather_api import WeatherObservation

logger = logging.getLogger(__name__)


@dataclass
class ComfortFeatures:
    """쾌적도 관련 피처."""

    # 기본 쾌적도 지수
    heat_index: float  # 체감온도 (온도 + 습도)
    wind_chill: float  # 바람 체감온도
    apparent_temperature: float  # 실제 체감온도

    # 불쾌감 지수
    discomfort_index: float  # THI (Temperature-Humidity Index)
    humidity_comfort: float  # 습도 쾌적도 (40-60% 최적)
    temperature_comfort: float  # 온도 쾌적도 (18-24°C 최적)

    # 상황별 쾌적도
    walking_comfort: float  # 도보 이동 쾌적도
    waiting_comfort: float  # 대기 시간 쾌적도
    clothing_index: float  # 의복 지수


@dataclass
class TemporalFeatures:
    """시간적 피처."""

    # 시간 주기
    hour_sin: float
    hour_cos: float
    day_of_week: int
    day_of_year_sin: float
    day_of_year_cos: float
    month: int

    # 출퇴근 시간대
    is_morning_commute: bool  # 7-9시
    is_evening_commute: bool  # 17-19시
    is_rush_hour: bool
    is_weekend: bool
    is_holiday: bool  # 간단한 공휴일 판별

    # 시간 기반 가중치
    commute_weight: float  # 출퇴근 시간 가중치
    time_of_day_weight: float  # 하루 중 시간대 가중치


@dataclass
class WeatherFeatures:
    """날씨 관련 피처."""

    # 기본 날씨 값 (정규화 포함)
    temperature_normalized: float  # -20~40°C를 0~1로
    humidity_normalized: float  # 0~100%를 0~1로
    wind_speed_normalized: float  # 0~30m/s를 0~1로
    precipitation_log: float  # log(1 + precipitation)

    # 날씨 강도 분류
    temperature_category: int  # 매우추움(0), 추움(1), 적정(2), 더움(3), 매우더움(4)
    humidity_category: int  # 건조(0), 적정(1), 습함(2)
    wind_category: int  # 무풍(0), 약함(1), 보통(2), 강함(3)
    precipitation_category: int  # 없음(0), 약함(1), 보통(2), 강함(3)

    # 날씨 조합 상태
    weather_severity_score: float  # 전체적인 날씨 심각도
    is_extreme_weather: bool  # 극한 날씨 여부
    is_pleasant_weather: bool  # 쾌적한 날씨 여부


@dataclass
class InteractionFeatures:
    """상호작용 피처."""

    # 온도-습도 상호작용
    temp_humidity_interaction: float
    heat_index_diff: float  # heat_index - temperature

    # 바람-온도 상호작용
    wind_temp_interaction: float
    wind_cooling_effect: float

    # 강수-온도 상호작용
    precipitation_temp_interaction: float
    snow_probability: float  # 눈 올 확률

    # 복합 지수
    weather_stress_index: float  # 날씨 스트레스 지수
    outdoor_activity_index: float  # 야외활동 적합성 지수


class WeatherFeatureEngineer:
    """실제 KMA 데이터 기반 피처 엔지니어링."""

    def __init__(self):
        # 정규화 범위
        self.temp_range = (-20, 40)  # 한국 기온 범위
        self.humidity_range = (0, 100)
        self.wind_range = (0, 30)  # m/s

        # 쾌적도 기준
        self.optimal_temp_range = (18, 24)
        self.optimal_humidity_range = (40, 60)
        self.optimal_wind_range = (0, 3)

        # 공휴일 (간단한 버전)
        self.holidays = self._get_korean_holidays()

    def engineer_features(
        self,
        observations: List[WeatherObservation],
        target_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """전체 피처 엔지니어링 수행."""

        if not observations:
            raise ValueError("No observations provided")

        # 최신 관측값 사용
        latest_obs = observations[-1]
        target_time = target_time or latest_obs.timestamp

        logger.info(f"Engineering features for {len(observations)} observations")

        # 각 피처 그룹 생성
        comfort_features = self._create_comfort_features(latest_obs)
        temporal_features = self._create_temporal_features(target_time)
        weather_features = self._create_weather_features(latest_obs)
        interaction_features = self._create_interaction_features(latest_obs, comfort_features)

        # 시계열 기반 피처 (여러 관측값 사용)
        trend_features = self._create_trend_features(observations)

        # 모든 피처 통합
        all_features = {
            **self._dataclass_to_dict(comfort_features, "comfort"),
            **self._dataclass_to_dict(temporal_features, "temporal"),
            **self._dataclass_to_dict(weather_features, "weather"),
            **self._dataclass_to_dict(interaction_features, "interaction"),
            **trend_features
        }

        # 피처 품질 점수 계산
        feature_quality = self._calculate_feature_quality(latest_obs, all_features)
        all_features["feature_quality_score"] = feature_quality

        logger.info(f"Generated {len(all_features)} features with quality score: {feature_quality:.3f}")

        return all_features

    def _create_comfort_features(self, obs: WeatherObservation) -> ComfortFeatures:
        """쾌적도 관련 피처 생성."""

        temp = obs.temperature_c
        humidity = obs.relative_humidity or 50.0  # 기본값
        wind_speed = obs.wind_speed_ms

        # 체감온도 계산 (Heat Index)
        heat_index = self._calculate_heat_index(temp, humidity)

        # 바람 체감온도 계산 (Wind Chill)
        wind_chill = self._calculate_wind_chill(temp, wind_speed)

        # 실제 체감온도 (복합 지수)
        apparent_temp = self._calculate_apparent_temperature(temp, humidity, wind_speed)

        # 불쾌감 지수 (THI)
        discomfort_index = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3

        # 개별 요소 쾌적도
        humidity_comfort = self._calculate_humidity_comfort(humidity)
        temperature_comfort = self._calculate_temperature_comfort(temp)

        # 상황별 쾌적도
        walking_comfort = self._calculate_walking_comfort(temp, humidity, wind_speed, obs.precipitation_mm)
        waiting_comfort = self._calculate_waiting_comfort(temp, humidity, wind_speed, obs.precipitation_mm)
        clothing_index = self._calculate_clothing_index(apparent_temp, wind_speed)

        return ComfortFeatures(
            heat_index=heat_index,
            wind_chill=wind_chill,
            apparent_temperature=apparent_temp,
            discomfort_index=discomfort_index,
            humidity_comfort=humidity_comfort,
            temperature_comfort=temperature_comfort,
            walking_comfort=walking_comfort,
            waiting_comfort=waiting_comfort,
            clothing_index=clothing_index
        )

    def _create_temporal_features(self, timestamp: datetime) -> TemporalFeatures:
        """시간적 피처 생성."""

        # 한국 시간으로 변환
        kst = timestamp.replace(tzinfo=ZoneInfo("Asia/Seoul"))

        # 시간 주기 인코딩
        hour = kst.hour
        day_of_week = kst.weekday()  # 0=월요일
        day_of_year = kst.timetuple().tm_yday
        month = kst.month

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_of_year_sin = math.sin(2 * math.pi * day_of_year / 365)
        day_of_year_cos = math.cos(2 * math.pi * day_of_year / 365)

        # 출퇴근 시간대 판별
        is_morning_commute = 7 <= hour <= 9
        is_evening_commute = 17 <= hour <= 19
        is_rush_hour = is_morning_commute or is_evening_commute
        is_weekend = day_of_week >= 5
        is_holiday = self._is_holiday(kst)

        # 가중치 계산
        commute_weight = 1.5 if is_rush_hour else 1.0
        time_of_day_weight = self._calculate_time_weight(hour)

        return TemporalFeatures(
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            day_of_week=day_of_week,
            day_of_year_sin=day_of_year_sin,
            day_of_year_cos=day_of_year_cos,
            month=month,
            is_morning_commute=is_morning_commute,
            is_evening_commute=is_evening_commute,
            is_rush_hour=is_rush_hour,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            commute_weight=commute_weight,
            time_of_day_weight=time_of_day_weight
        )

    def _create_weather_features(self, obs: WeatherObservation) -> WeatherFeatures:
        """날씨 관련 피처 생성."""

        temp = obs.temperature_c
        humidity = obs.relative_humidity or 50.0
        wind_speed = obs.wind_speed_ms
        precipitation = obs.precipitation_mm

        # 정규화
        temp_norm = self._normalize_value(temp, self.temp_range)
        humidity_norm = self._normalize_value(humidity, self.humidity_range)
        wind_norm = self._normalize_value(wind_speed, self.wind_range)
        precip_log = math.log1p(precipitation)  # log(1 + x)

        # 카테고리 분류
        temp_category = self._categorize_temperature(temp)
        humidity_category = self._categorize_humidity(humidity)
        wind_category = self._categorize_wind(wind_speed)
        precip_category = self._categorize_precipitation(precipitation)

        # 날씨 심각도 점수
        weather_severity = self._calculate_weather_severity(temp, humidity, wind_speed, precipitation)

        # 극한/쾌적 날씨 판별
        is_extreme = weather_severity > 0.7
        is_pleasant = 0.2 <= weather_severity <= 0.4

        return WeatherFeatures(
            temperature_normalized=temp_norm,
            humidity_normalized=humidity_norm,
            wind_speed_normalized=wind_norm,
            precipitation_log=precip_log,
            temperature_category=temp_category,
            humidity_category=humidity_category,
            wind_category=wind_category,
            precipitation_category=precip_category,
            weather_severity_score=weather_severity,
            is_extreme_weather=is_extreme,
            is_pleasant_weather=is_pleasant
        )

    def _create_interaction_features(self, obs: WeatherObservation, comfort: ComfortFeatures) -> InteractionFeatures:
        """상호작용 피처 생성."""

        temp = obs.temperature_c
        humidity = obs.relative_humidity or 50.0
        wind_speed = obs.wind_speed_ms
        precipitation = obs.precipitation_mm

        # 온도-습도 상호작용
        temp_humidity_interaction = temp * humidity / 100
        heat_index_diff = comfort.heat_index - temp

        # 바람-온도 상호작용
        wind_temp_interaction = wind_speed * abs(temp - 20)  # 20°C 기준
        wind_cooling_effect = wind_speed * max(0, temp - 25)  # 25°C 이상에서 바람 효과

        # 강수-온도 상호작용
        precip_temp_interaction = precipitation * abs(temp - 15)
        snow_probability = self._calculate_snow_probability(temp, precipitation, humidity)

        # 복합 지수
        weather_stress = self._calculate_weather_stress_index(temp, humidity, wind_speed, precipitation)
        outdoor_activity = self._calculate_outdoor_activity_index(comfort, obs)

        return InteractionFeatures(
            temp_humidity_interaction=temp_humidity_interaction,
            heat_index_diff=heat_index_diff,
            wind_temp_interaction=wind_temp_interaction,
            wind_cooling_effect=wind_cooling_effect,
            precipitation_temp_interaction=precip_temp_interaction,
            snow_probability=snow_probability,
            weather_stress_index=weather_stress,
            outdoor_activity_index=outdoor_activity
        )

    def _create_trend_features(self, observations: List[WeatherObservation]) -> Dict[str, float]:
        """시계열 트렌드 피처 생성."""

        if len(observations) < 2:
            return {
                "temp_trend": 0.0,
                "temp_volatility": 0.0,
                "humidity_trend": 0.0,
                "wind_trend": 0.0,
                "pressure_tendency": 0.0,
                "weather_stability": 0.5
            }

        # 시간순 정렬
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)

        # 온도 트렌드
        temps = [obs.temperature_c for obs in sorted_obs]
        temp_trend = self._calculate_trend(temps)
        temp_volatility = np.std(temps) if len(temps) > 1 else 0.0

        # 습도 트렌드
        humidities = [obs.relative_humidity or 50.0 for obs in sorted_obs]
        humidity_trend = self._calculate_trend(humidities)

        # 바람 트렌드
        winds = [obs.wind_speed_ms for obs in sorted_obs]
        wind_trend = self._calculate_trend(winds)

        # 기압 경향 (간접적으로 추정)
        pressure_tendency = self._estimate_pressure_tendency(sorted_obs)

        # 날씨 안정성
        weather_stability = self._calculate_weather_stability(sorted_obs)

        return {
            "temp_trend": temp_trend,
            "temp_volatility": temp_volatility,
            "humidity_trend": humidity_trend,
            "wind_trend": wind_trend,
            "pressure_tendency": pressure_tendency,
            "weather_stability": weather_stability
        }

    # 헬퍼 메서드들

    def _calculate_heat_index(self, temp: float, humidity: float) -> float:
        """체감온도 계산 (Heat Index)."""
        if temp < 27 or humidity < 40:
            return temp

        # Rothfusz regression
        T = temp
        RH = humidity

        HI = (-8.78469475556 + 1.61139411 * T + 2.33854883889 * RH +
              -0.14611605 * T * RH + -0.012308094 * T * T +
              -0.0164248277778 * RH * RH + 0.002211732 * T * T * RH +
              0.00072546 * T * RH * RH + -0.000003582 * T * T * RH * RH)

        return HI

    def _calculate_wind_chill(self, temp: float, wind_speed: float) -> float:
        """바람 체감온도 계산."""
        if temp > 10 or wind_speed < 1.34:
            return temp

        # 풍속을 km/h로 변환
        wind_kmh = wind_speed * 3.6

        wind_chill = (13.12 + 0.6215 * temp - 11.37 * (wind_kmh ** 0.16) +
                     0.3965 * temp * (wind_kmh ** 0.16))

        return wind_chill

    def _calculate_apparent_temperature(self, temp: float, humidity: float, wind_speed: float) -> float:
        """실제 체감온도 계산."""
        # 온도에 따라 다른 공식 사용
        if temp >= 27 and humidity >= 40:
            return self._calculate_heat_index(temp, humidity)
        elif temp <= 10 and wind_speed >= 1.34:
            return self._calculate_wind_chill(temp, wind_speed)
        else:
            # 일반적인 체감온도 공식
            return temp + 0.33 * (humidity / 100 * 6.105 * math.exp(17.27 * temp / (237.7 + temp))) - 0.7 * wind_speed - 4.0

    def _calculate_humidity_comfort(self, humidity: float) -> float:
        """습도 쾌적도 계산 (40-60% 최적)."""
        if 40 <= humidity <= 60:
            return 1.0
        elif humidity < 40:
            return max(0, 1 - (40 - humidity) / 40)
        else:
            return max(0, 1 - (humidity - 60) / 40)

    def _calculate_temperature_comfort(self, temp: float) -> float:
        """온도 쾌적도 계산 (18-24°C 최적)."""
        if 18 <= temp <= 24:
            return 1.0
        elif temp < 18:
            return max(0, 1 - (18 - temp) / 20)
        else:
            return max(0, 1 - (temp - 24) / 20)

    def _calculate_walking_comfort(self, temp: float, humidity: float, wind: float, precip: float) -> float:
        """도보 이동 쾌적도."""
        base_score = 0.8

        # 온도 패널티
        if temp < 0:
            base_score -= 0.3
        elif temp > 30:
            base_score -= 0.2

        # 강수 패널티
        if precip > 0:
            base_score -= min(0.4, precip * 0.1)

        # 바람 효과 (적당한 바람은 도움)
        if wind > 5:
            base_score -= 0.1
        elif 1 < wind < 3 and temp > 25:
            base_score += 0.1

        return max(0, min(1, base_score))

    def _calculate_waiting_comfort(self, temp: float, humidity: float, wind: float, precip: float) -> float:
        """대기 시간 쾌적도."""
        base_score = 0.7

        # 온도 패널티 (대기 시에는 더 민감)
        if temp < 5:
            base_score -= 0.4
        elif temp > 28:
            base_score -= 0.3

        # 강수 패널티 (대기 시에는 더 심각)
        if precip > 0:
            base_score -= min(0.5, precip * 0.15)

        # 바람 패널티 (대기 시에는 바람이 불편)
        if wind > 3:
            base_score -= min(0.2, wind * 0.05)

        return max(0, min(1, base_score))

    def _calculate_clothing_index(self, apparent_temp: float, wind_speed: float) -> float:
        """의복 지수 계산."""
        # 체감온도 기반 의복 지수 (0: 매우 두껍게, 1: 매우 얇게)
        if apparent_temp < 0:
            clothing = 0.0
        elif apparent_temp < 10:
            clothing = 0.2
        elif apparent_temp < 20:
            clothing = 0.5
        elif apparent_temp < 25:
            clothing = 0.7
        else:
            clothing = 1.0

        # 바람 보정
        if wind_speed > 5:
            clothing = max(0, clothing - 0.1)

        return clothing

    def _normalize_value(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """값을 0-1 범위로 정규화."""
        min_val, max_val = range_tuple
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def _categorize_temperature(self, temp: float) -> int:
        """온도 카테고리 분류."""
        if temp < 0:
            return 0  # 매우 추움
        elif temp < 10:
            return 1  # 추움
        elif temp < 28:
            return 2  # 적정
        elif temp < 35:
            return 3  # 더움
        else:
            return 4  # 매우 더움

    def _categorize_humidity(self, humidity: float) -> int:
        """습도 카테고리 분류."""
        if humidity < 30:
            return 0  # 건조
        elif humidity < 70:
            return 1  # 적정
        else:
            return 2  # 습함

    def _categorize_wind(self, wind: float) -> int:
        """바람 카테고리 분류."""
        if wind < 1:
            return 0  # 무풍
        elif wind < 3:
            return 1  # 약함
        elif wind < 8:
            return 2  # 보통
        else:
            return 3  # 강함

    def _categorize_precipitation(self, precip: float) -> int:
        """강수 카테고리 분류."""
        if precip == 0:
            return 0  # 없음
        elif precip < 1:
            return 1  # 약함
        elif precip < 10:
            return 2  # 보통
        else:
            return 3  # 강함

    def _calculate_weather_severity(self, temp: float, humidity: float, wind: float, precip: float) -> float:
        """전체적인 날씨 심각도 계산."""
        severity = 0.0

        # 온도 심각도
        if temp < -10 or temp > 35:
            severity += 0.4
        elif temp < 0 or temp > 30:
            severity += 0.2

        # 습도 심각도
        if humidity < 20 or humidity > 80:
            severity += 0.2

        # 바람 심각도
        if wind > 10:
            severity += 0.3
        elif wind > 5:
            severity += 0.1

        # 강수 심각도
        if precip > 10:
            severity += 0.4
        elif precip > 0:
            severity += 0.1

        return min(1.0, severity)

    def _calculate_snow_probability(self, temp: float, precip: float, humidity: float) -> float:
        """눈 올 확률 계산."""
        if precip == 0:
            return 0.0

        if temp <= 0:
            return 0.9
        elif temp <= 2:
            return 0.7
        elif temp <= 4:
            return 0.3
        else:
            return 0.0

    def _calculate_weather_stress_index(self, temp: float, humidity: float, wind: float, precip: float) -> float:
        """날씨 스트레스 지수."""
        stress = 0.0

        # 각 요소별 스트레스 계산
        temp_stress = max(0, abs(temp - 20) / 20)  # 20°C 기준
        humidity_stress = max(0, abs(humidity - 50) / 50)  # 50% 기준
        wind_stress = min(1, wind / 10)  # 10m/s 기준
        precip_stress = min(1, precip / 10)  # 10mm 기준

        # 가중 평균
        stress = (0.3 * temp_stress + 0.2 * humidity_stress +
                 0.2 * wind_stress + 0.3 * precip_stress)

        return stress

    def _calculate_outdoor_activity_index(self, comfort: ComfortFeatures, obs: WeatherObservation) -> float:
        """야외활동 적합성 지수."""
        base_score = (comfort.temperature_comfort + comfort.humidity_comfort) / 2

        # 강수 패널티
        if obs.precipitation_mm > 0:
            base_score *= 0.5

        # 바람 보정
        if obs.wind_speed_ms > 5:
            base_score *= 0.8

        return max(0, min(1, base_score))

    def _calculate_trend(self, values: List[float]) -> float:
        """값들의 트렌드 계산 (선형 회귀 기울기)."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # 선형 회귀 기울기 계산
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _estimate_pressure_tendency(self, observations: List[WeatherObservation]) -> float:
        """기압 경향 간접 추정."""
        # 온도와 습도 변화를 통해 기압 변화 추정
        if len(observations) < 3:
            return 0.0

        temps = [obs.temperature_c for obs in observations[-3:]]
        humids = [obs.relative_humidity or 50.0 for obs in observations[-3:]]

        temp_change = temps[-1] - temps[0]
        humid_change = humids[-1] - humids[0]

        # 간단한 경험적 공식
        pressure_tendency = -(temp_change * 0.1 + humid_change * 0.05)

        return max(-1, min(1, pressure_tendency))

    def _calculate_weather_stability(self, observations: List[WeatherObservation]) -> float:
        """날씨 안정성 계산."""
        if len(observations) < 2:
            return 0.5

        # 각 요소별 변동성 계산
        temps = [obs.temperature_c for obs in observations]
        winds = [obs.wind_speed_ms for obs in observations]
        precips = [obs.precipitation_mm for obs in observations]

        temp_stability = 1.0 - min(1.0, np.std(temps) / 10)
        wind_stability = 1.0 - min(1.0, np.std(winds) / 5)
        precip_stability = 1.0 - min(1.0, np.std(precips) / 5)

        # 평균 안정성
        stability = (temp_stability + wind_stability + precip_stability) / 3

        return stability

    def _calculate_feature_quality(self, obs: WeatherObservation, features: Dict[str, Any]) -> float:
        """피처 품질 점수 계산."""
        quality = 1.0

        # 데이터 완성도 확인
        if obs.relative_humidity is None:
            quality -= 0.2

        if obs.temperature_c == 0.0:  # 의심스러운 0도
            quality -= 0.1

        # 극값 확인
        if abs(obs.temperature_c) > 40:
            quality -= 0.1

        if obs.wind_speed_ms > 30:
            quality -= 0.1

        # 피처 일관성 확인
        try:
            if features.get("weather_weather_severity_score", 0) > 0.8:
                quality -= 0.1  # 극한 날씨는 품질 감소
        except:
            pass

        return max(0.0, min(1.0, quality))

    def _calculate_time_weight(self, hour: int) -> float:
        """시간대별 가중치 계산."""
        # 출퇴근 시간에 더 높은 가중치
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 1.5
        elif 6 <= hour <= 10 or 16 <= hour <= 20:
            return 1.2
        else:
            return 1.0

    def _get_korean_holidays(self) -> List[str]:
        """한국 공휴일 목록 (간단한 버전)."""
        return [
            "01-01",  # 신정
            "02-11", "02-12", "02-13",  # 설날 (대략)
            "03-01",  # 삼일절
            "05-05",  # 어린이날
            "06-06",  # 현충일
            "08-15",  # 광복절
            "10-03",  # 개천절
            "10-09",  # 한글날
            "12-25",  # 크리스마스
        ]

    def _is_holiday(self, dt: datetime) -> bool:
        """공휴일 여부 확인."""
        date_str = dt.strftime("%m-%d")
        return date_str in self.holidays

    def _dataclass_to_dict(self, dc, prefix: str) -> Dict[str, Any]:
        """데이터클래스를 딕셔너리로 변환."""
        result = {}
        for field_name, field_value in dc.__dict__.items():
            key = f"{prefix}_{field_name}"
            result[key] = field_value
        return result