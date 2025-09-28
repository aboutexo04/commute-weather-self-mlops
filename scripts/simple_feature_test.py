#!/usr/bin/env python3
"""간단한 피처 엔지니어링 테스트 (의존성 최소화)."""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

def create_mock_weather_observation():
    """테스트용 모의 기상 관측 데이터 생성."""
    class MockWeatherObservation:
        def __init__(self):
            self.temperature = 22.5  # 22.5°C
            self.humidity = 65  # 65%
            self.wind_speed = 3.2  # 3.2 m/s
            self.precipitation = 0.0  # 0.0 mm
            self.pressure = 1013.2  # 1013.2 hPa
            self.visibility = 15000  # 15km
            self.wind_direction = 180  # 남풍
            self.cloud_coverage = 30  # 30%
            self.observation_time = datetime.now()
            self.station_id = "108"  # 서울

    return MockWeatherObservation()

def test_basic_feature_calculations():
    """기본 피처 계산 로직 테스트."""
    print("🧮 기본 피처 계산 테스트")
    print("-" * 40)

    obs = create_mock_weather_observation()

    # 1. 체감온도 계산 (Heat Index)
    temp_c = obs.temperature
    humidity = obs.humidity

    # Heat Index 공식 (화씨로 변환 후 계산)
    temp_f = temp_c * 9/5 + 32
    hi_f = (
        -42.379 +
        2.04901523 * temp_f +
        10.14333127 * humidity -
        0.22475541 * temp_f * humidity -
        6.83783e-3 * temp_f**2 -
        5.481717e-2 * humidity**2 +
        1.22874e-3 * temp_f**2 * humidity +
        8.5282e-4 * temp_f * humidity**2 -
        1.99e-6 * temp_f**2 * humidity**2
    )
    heat_index = (hi_f - 32) * 5/9  # 섭씨로 변환

    print(f"입력 데이터:")
    print(f"  온도: {temp_c}°C")
    print(f"  습도: {humidity}%")
    print(f"  풍속: {obs.wind_speed}m/s")
    print(f"계산 결과:")
    print(f"  체감온도(Heat Index): {heat_index:.1f}°C")

    # 2. 바람 체감온도 (Wind Chill) - 낮은 온도에서만 의미
    if temp_c < 10:
        wind_chill = (
            13.12 + 0.6215 * temp_c -
            11.37 * (obs.wind_speed * 3.6)**0.16 +
            0.3965 * temp_c * (obs.wind_speed * 3.6)**0.16
        )
        print(f"  바람 체감온도: {wind_chill:.1f}°C")
    else:
        print(f"  바람 체감온도: N/A (온도가 높음)")

    # 3. 불쾌감 지수 (THI)
    thi = 0.81 * temp_c + 0.01 * humidity * (0.99 * temp_c - 14.3) + 46.3
    print(f"  불쾌감 지수(THI): {thi:.1f}")

    # 4. 쾌적도 점수 (0-100)
    temp_comfort = max(0, 100 - abs(temp_c - 21) * 5)  # 21°C 기준
    humidity_comfort = max(0, 100 - abs(humidity - 50) * 2)  # 50% 기준
    wind_comfort = max(0, 100 - obs.wind_speed * 10)  # 낮은 풍속이 좋음

    overall_comfort = (temp_comfort + humidity_comfort + wind_comfort) / 3
    print(f"  전체 쾌적도: {overall_comfort:.1f}/100")

    # 5. 시간적 피처
    now = datetime.now()
    hour_sin = round(sin_cos_encode(now.hour, 24)[0], 3)
    hour_cos = round(sin_cos_encode(now.hour, 24)[1], 3)

    print(f"시간적 피처:")
    print(f"  현재 시간: {now.hour}시")
    print(f"  시간 sin 인코딩: {hour_sin}")
    print(f"  시간 cos 인코딩: {hour_cos}")

    return True

def sin_cos_encode(value, max_value):
    """순환 특성을 가진 값을 sin/cos로 인코딩."""
    import math
    radians = 2 * math.pi * value / max_value
    return math.sin(radians), math.cos(radians)

def test_feature_engineering_integration():
    """피처 엔지니어링 모듈 통합 테스트."""
    print("\n🔗 피처 엔지니어링 모듈 통합 테스트")
    print("-" * 40)

    try:
        # 피처 엔지니어링 모듈 임포트 시도
        from commute_weather.features.feature_engineering import WeatherFeatureEngineer
        print("✅ WeatherFeatureEngineer 클래스 임포트 성공")

        # 인스턴스 생성 시도
        engineer = WeatherFeatureEngineer()
        print("✅ WeatherFeatureEngineer 인스턴스 생성 성공")

        # 기본 메서드 존재 확인
        methods = ['engineer_features', 'evaluate_feature_quality']
        for method in methods:
            if hasattr(engineer, method):
                print(f"✅ {method} 메서드 확인")
            else:
                print(f"❌ {method} 메서드 없음")

        return True

    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {str(e)}")
        print("   의존성 패키지가 설치되지 않았을 수 있습니다.")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        return False

def test_asos_collector_integration():
    """ASOS 수집기 통합 테스트."""
    print("\n📡 ASOS 수집기 통합 테스트")
    print("-" * 40)

    try:
        from commute_weather.data_sources.asos_collector import AsosCollector
        print("✅ AsosCollector 클래스 임포트 성공")

        collector = AsosCollector()
        print("✅ AsosCollector 인스턴스 생성 성공")

        # 기본 메서드 존재 확인
        methods = ['fetch_latest_observation', 'select_core_features']
        for method in methods:
            if hasattr(collector, method):
                print(f"✅ {method} 메서드 확인")
            else:
                print(f"❌ {method} 메서드 없음")

        return True

    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        return False

def main():
    """메인 테스트 실행."""
    print("🚀 피처 엔지니어링 시스템 간단 테스트")
    print("=" * 60)

    # 환경 체크
    print("🔧 환경 설정 확인...")
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print(f"✅ .env 파일 확인: {env_file}")
    else:
        print("⚠️ .env 파일이 없습니다.")

    print(f"✅ 프로젝트 루트: {project_root}")

    # 테스트 실행
    tests = [
        ("기본 피처 계산", test_basic_feature_calculations),
        ("피처 엔지니어링 모듈", test_feature_engineering_integration),
        ("ASOS 수집기 모듈", test_asos_collector_integration)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 실패: {str(e)}")
            results.append((test_name, False))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("-" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")

    print(f"\n전체 결과: {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n🎉 모든 테스트 통과!")
        print("\n다음 단계:")
        print("1. pip install -e . (의존성 설치 완료 후)")
        print("2. python scripts/test_feature_engineering.py (전체 테스트)")
        print("3. 실제 KMA API 연동 테스트")
    else:
        print(f"\n⚠️ {total-passed}개 테스트 실패")
        print("의존성 패키지 설치 후 재테스트 필요")

if __name__ == "__main__":
    main()