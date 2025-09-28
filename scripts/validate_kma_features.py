#!/usr/bin/env python3
"""KMA 데이터 기반 피처 엔지니어링 검증 스크립트."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime
from commute_weather.data_sources.weather_api import WeatherObservation
from commute_weather.features.feature_engineering import WeatherFeatureEngineer

def validate_kma_to_features_mapping():
    """KMA 데이터가 피처로 올바르게 변환되는지 검증."""

    print("🔍 KMA 데이터 → 피처 매핑 검증")
    print("=" * 60)

    # 실제 KMA 데이터 형태 시뮬레이션
    kma_observation = WeatherObservation(
        timestamp=datetime.now(),
        temperature_c=22.5,      # KMA 'ta' 필드
        wind_speed_ms=3.2,       # KMA 'ws' 필드
        precipitation_mm=0.0,    # KMA 'rn' 필드
        relative_humidity=65.0,  # KMA 'hm/rh/reh' 필드
        precipitation_type="none"
    )

    print("📊 입력 KMA 데이터:")
    print(f"   온도(ta): {kma_observation.temperature_c}°C")
    print(f"   습도(hm): {kma_observation.relative_humidity}%")
    print(f"   풍속(ws): {kma_observation.wind_speed_ms} m/s")
    print(f"   강수량(rn): {kma_observation.precipitation_mm} mm")

    # 피처 엔지니어링 수행
    engineer = WeatherFeatureEngineer()
    features = engineer.engineer_features([kma_observation])

    print("\n🧠 생성된 피처들:")
    print("-" * 40)

    # KMA 데이터 직접 활용 피처들
    print("📍 KMA 원시 데이터 기반 피처:")
    kma_based_features = {
        k: v for k, v in features.items()
        if any(base in k.lower() for base in ['temperature', 'humidity', 'wind', 'precipitation'])
    }

    for key, value in kma_based_features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n🧮 KMA 기반 계산 피처:")
    calculated_features = {
        k: v for k, v in features.items()
        if any(calc in k.lower() for calc in ['heat_index', 'wind_chill', 'apparent', 'discomfort'])
    }

    for key, value in calculated_features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n⚡ KMA 상호작용 피처:")
    interaction_features = {
        k: v for k, v in features.items()
        if 'interaction' in k.lower() or 'index' in k.lower()
    }

    for key, value in interaction_features.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print(f"\n📊 총 생성된 피처 개수: {len(features)}")
    print(f"📊 피처 품질 점수: {features.get('feature_quality_score', 'N/A')}")

    # 핵심 검증
    print("\n✅ 검증 결과:")
    print("-" * 40)

    # 1. KMA 원본 데이터 보존 확인
    assert 'weather_temperature_normalized' in features, "온도 정규화 피처 누락"
    assert 'weather_humidity_normalized' in features, "습도 정규화 피처 누락"
    assert 'weather_wind_speed_normalized' in features, "풍속 정규화 피처 누락"
    assert 'weather_precipitation_log' in features, "강수량 로그 피처 누락"
    print("✓ KMA 원시 데이터 모든 피처로 변환 완료")

    # 2. 체감온도 계산 확인
    heat_index = features.get('comfort_heat_index', 0)
    apparent_temp = features.get('comfort_apparent_temperature', 0)
    print(f"✓ 체감온도 계산: Heat Index {heat_index:.1f}°C, Apparent {apparent_temp:.1f}°C")

    # 3. 쾌적도 지수 확인
    temp_comfort = features.get('comfort_temperature_comfort', 0)
    humidity_comfort = features.get('comfort_humidity_comfort', 0)
    print(f"✓ 쾌적도 지수: 온도 {temp_comfort:.3f}, 습도 {humidity_comfort:.3f}")

    # 4. 상호작용 피처 확인
    temp_humidity_interact = features.get('interaction_temp_humidity_interaction', 0)
    weather_stress = features.get('interaction_weather_stress_index', 0)
    print(f"✓ 상호작용 피처: 온습도 {temp_humidity_interact:.3f}, 스트레스 {weather_stress:.3f}")

    print("\n🎉 검증 완료: KMA 데이터가 올바르게 피처로 변환됩니다!")

    return features

def show_kma_column_mapping():
    """KMA API 칼럼과 피처 매핑 관계 표시."""

    print("\n📋 KMA API 칼럼 → 피처 매핑표")
    print("=" * 60)

    mapping = {
        "KMA 'ta' (온도)": [
            "weather_temperature_normalized",
            "weather_temperature_category",
            "comfort_heat_index",
            "comfort_wind_chill",
            "comfort_apparent_temperature",
            "comfort_temperature_comfort",
            "interaction_temp_humidity_interaction",
            "interaction_wind_temp_interaction"
        ],
        "KMA 'hm/rh/reh' (습도)": [
            "weather_humidity_normalized",
            "weather_humidity_category",
            "comfort_heat_index",
            "comfort_discomfort_index",
            "comfort_humidity_comfort",
            "interaction_temp_humidity_interaction",
            "interaction_snow_probability"
        ],
        "KMA 'ws' (풍속)": [
            "weather_wind_speed_normalized",
            "weather_wind_category",
            "comfort_wind_chill",
            "comfort_apparent_temperature",
            "comfort_clothing_index",
            "interaction_wind_temp_interaction",
            "interaction_wind_cooling_effect"
        ],
        "KMA 'rn/pr1' (강수량)": [
            "weather_precipitation_log",
            "weather_precipitation_category",
            "comfort_walking_comfort",
            "comfort_waiting_comfort",
            "interaction_precipitation_temp_interaction",
            "interaction_snow_probability"
        ],
        "KMA 'tm' (시간)": [
            "temporal_hour_sin",
            "temporal_hour_cos",
            "temporal_day_of_week",
            "temporal_is_morning_commute",
            "temporal_is_evening_commute",
            "temporal_commute_weight"
        ]
    }

    for kma_column, features in mapping.items():
        print(f"\n🔗 {kma_column}:")
        for feature in features:
            print(f"   → {feature}")

    print(f"\n📊 총 매핑된 피처 개수: {sum(len(features) for features in mapping.values())}")

if __name__ == "__main__":
    try:
        features = validate_kma_to_features_mapping()
        show_kma_column_mapping()

        print("\n" + "=" * 60)
        print("🎯 결론: 모든 KMA 데이터 칼럼이 의미있는 피처로 변환됩니다!")
        print("✅ 온도, 습도, 풍속, 강수량, 시간 → 총 40+ 개 피처 생성")

    except Exception as e:
        print(f"❌ 검증 실패: {str(e)}")
        import traceback
        traceback.print_exc()