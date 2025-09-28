#!/usr/bin/env python3
"""실제 KMA 데이터를 사용한 피처 엔지니어링 시스템 테스트 스크립트."""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from commute_weather.data_sources.asos_collector import AsosCollector
from commute_weather.features.feature_engineering import WeatherFeatureEngineer
from commute_weather.storage.s3_manager import S3StorageManager
from commute_weather.monitoring import get_logger


def test_feature_engineering_with_real_data():
    """실제 KMA 데이터로 피처 엔지니어링 테스트."""
    logger = get_logger("test_feature_engineering")

    print("🧪 KMA 데이터 피처 엔지니어링 테스트 시작")
    print("=" * 60)

    # 1. 환경 변수 확인
    auth_key = os.getenv('KMA_AUTH_KEY')
    if not auth_key:
        print("❌ KMA_AUTH_KEY가 설정되지 않았습니다.")
        return False

    station_id = os.getenv('KMA_STATION_ID', '108')  # 서울 기본값

    print(f"📡 KMA 인증키: {'*' * (len(auth_key) - 4)}{auth_key[-4:]}")
    print(f"🏢 관측소 ID: {station_id}")

    try:
        # 2. ASOS 데이터 수집기 초기화
        asos_collector = AsosCollector()
        print("\n🔄 ASOS 데이터 수집 중...")

        # 최신 관측 데이터 가져오기
        observation = asos_collector.fetch_latest_observation(
            station_id=station_id,
            timeout=30
        )

        if not observation:
            print("❌ 관측 데이터를 가져올 수 없습니다.")
            return False

        print(f"✅ 관측 데이터 수집 완료: {observation.observation_time}")
        print(f"   온도: {observation.temperature}°C")
        print(f"   습도: {observation.humidity}%")
        print(f"   풍속: {observation.wind_speed}m/s")
        print(f"   강수량: {observation.precipitation}mm")

        # 3. 핵심 피처 선택
        print("\n🎯 핵심 피처 선택 중...")
        core_features = asos_collector.select_core_features(
            observation=observation,
            min_quality_threshold=50.0  # 테스트를 위해 낮은 임계값 사용
        )

        print(f"✅ 핵심 피처 {len(core_features)}개 선택")
        for feature_name, value in core_features.items():
            print(f"   {feature_name}: {value}")

        # 4. 피처 엔지니어링 실행
        print("\n⚙️ 피처 엔지니어링 실행 중...")
        feature_engineer = WeatherFeatureEngineer()

        # WeatherObservation 객체를 리스트로 변환 (최소 1개 필요)
        observations = [observation]

        # 출퇴근 시간 시뮬레이션
        target_times = [
            datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),  # 출근 시간
            datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)  # 퇴근 시간
        ]

        results = {}
        for time_label, target_time in zip(['morning_commute', 'evening_commute'], target_times):
            print(f"\n🕐 {time_label} ({target_time.strftime('%H:%M')}) 피처 생성 중...")

            engineered_features = feature_engineer.engineer_features(
                observations=observations,
                target_time=target_time
            )

            results[time_label] = engineered_features

            print(f"✅ {len(engineered_features)}개 피처 생성 완료")

            # 주요 피처들 출력
            print("   주요 피처:")
            key_features = [
                'comfort_score', 'heat_index', 'wind_chill', 'visibility_impact',
                'precipitation_severity', 'commute_difficulty', 'weather_trend'
            ]
            for feature in key_features:
                if feature in engineered_features:
                    value = engineered_features[feature]
                    print(f"     {feature}: {value}")

        # 5. 피처 품질 평가
        print("\n📊 피처 품질 평가...")
        quality_scores = feature_engineer.evaluate_feature_quality(observations)

        print(f"✅ 전체 피처 품질 점수: {quality_scores.get('overall_quality', 0):.1f}/100")

        feature_quality = quality_scores.get('feature_quality', {})
        if feature_quality:
            print("   개별 피처 품질:")
            for feature, score in sorted(feature_quality.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     {feature}: {score:.1f}")

        # 6. S3에 결과 저장 (옵션)
        if os.getenv('COMMUTE_S3_BUCKET'):
            print("\n💾 S3에 테스트 결과 저장 중...")
            try:
                s3_manager = S3StorageManager()

                test_results = {
                    'timestamp': datetime.now().isoformat(),
                    'station_id': station_id,
                    'raw_observation': {
                        'temperature': observation.temperature,
                        'humidity': observation.humidity,
                        'wind_speed': observation.wind_speed,
                        'precipitation': observation.precipitation,
                        'observation_time': observation.observation_time.isoformat()
                    },
                    'core_features': core_features,
                    'engineered_features': results,
                    'quality_scores': quality_scores
                }

                # features/test/ 디렉토리에 저장
                key = f"features/test/feature_engineering_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                success = s3_manager.save_features(
                    features=test_results,
                    key=key
                )

                if success:
                    print(f"✅ S3에 저장 완료: {key}")
                else:
                    print("⚠️ S3 저장 실패")

            except Exception as e:
                print(f"⚠️ S3 저장 중 오류: {str(e)}")

        # 7. 테스트 결과 요약
        print("\n" + "=" * 60)
        print("🎉 피처 엔지니어링 테스트 완료!")
        print(f"✅ 원본 관측 데이터: 수집 성공")
        print(f"✅ 핵심 피처 선택: {len(core_features)}개")
        print(f"✅ 출근시간 피처: {len(results['morning_commute'])}개")
        print(f"✅ 퇴근시간 피처: {len(results['evening_commute'])}개")
        print(f"✅ 피처 품질 점수: {quality_scores.get('overall_quality', 0):.1f}/100")

        # 생성된 피처 카테고리별 요약
        sample_features = results['morning_commute']
        categories = {
            'comfort': [k for k in sample_features.keys() if 'comfort' in k or 'heat_index' in k or 'wind_chill' in k],
            'temporal': [k for k in sample_features.keys() if 'hour' in k or 'season' in k or 'weekend' in k],
            'weather': [k for k in sample_features.keys() if 'temp' in k or 'wind' in k or 'precip' in k or 'humid' in k],
            'interaction': [k for k in sample_features.keys() if 'interaction' in k or 'difficulty' in k or 'trend' in k]
        }

        print("\n📈 생성된 피처 카테고리:")
        for category, features in categories.items():
            if features:
                print(f"   {category.capitalize()}: {len(features)}개 피처")

        return True

    except Exception as e:
        logger.error(f"피처 엔지니어링 테스트 실패: {str(e)}")
        print(f"\n❌ 테스트 실패: {str(e)}")
        return False


def test_feature_pipeline_integration():
    """피처 엔지니어링과 MLOps 파이프라인 통합 테스트."""
    print("\n🔗 MLOps 파이프라인 통합 테스트...")

    try:
        # realtime_mlops_pipeline 테스트 (간단한 호출)
        from commute_weather.pipelines.realtime_mlops_pipeline import realtime_mlops_flow

        print("✅ MLOps 파이프라인 모듈 임포트 성공")
        print("   - 실제 파이프라인 실행은 별도로 수행해주세요.")
        print("   - 명령어: python scripts/run_realtime_mlops.py")

        return True

    except ImportError as e:
        print(f"⚠️ MLOps 파이프라인 모듈 임포트 실패: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 KMA 데이터 피처 엔지니어링 종합 테스트")
    print("=" * 80)

    # 환경 체크
    print("🔧 환경 설정 확인...")
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print(f"✅ .env 파일 확인: {env_file}")
        # .env 파일 로드
        from dotenv import load_dotenv
        load_dotenv(env_file)
    else:
        print("⚠️ .env 파일이 없습니다. 환경 변수를 직접 설정해주세요.")

    # 메인 테스트 실행
    success = test_feature_engineering_with_real_data()

    if success:
        print("\n" + "=" * 80)
        test_feature_pipeline_integration()

        print("\n🎊 모든 테스트 완료!")
        print("\n다음 단계:")
        print("1. python scripts/run_realtime_mlops.py - 전체 MLOps 파이프라인 실행")
        print("2. python scripts/deploy_mlops.py run - MLOps 애플리케이션 실행")
        print("3. http://localhost:8000/mlops/health - 시스템 상태 확인")
    else:
        print("\n❌ 테스트 실패. 오류를 확인하고 다시 시도해주세요.")
        sys.exit(1)