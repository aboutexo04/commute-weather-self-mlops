#!/usr/bin/env python3
"""향상된 모델 훈련 실행 스크립트."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from commute_weather.ml.enhanced_training import EnhancedCommutePredictor, EnhancedTrainingConfig
from commute_weather.ml.mlflow_integration import MLflowManager
from commute_weather.data_sources.weather_api import WeatherObservation
from commute_weather.storage.s3_manager import S3StorageManager
from commute_weather.config import ProjectPaths

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_observations() -> List[List[WeatherObservation]]:
    """샘플 관측 데이터 생성 (테스트용)"""

    logger.info("Creating sample weather observations")

    observations_sets = []
    base_time = datetime.now()

    # 다양한 날씨 시나리오 생성
    scenarios = [
        # 쾌적한 날씨
        {"temp": 22, "humidity": 55, "wind": 2.0, "precip": 0.0},
        # 더운 날씨
        {"temp": 32, "humidity": 75, "wind": 1.0, "precip": 0.0},
        # 추운 날씨
        {"temp": -5, "humidity": 45, "wind": 8.0, "precip": 0.0},
        # 비 오는 날씨
        {"temp": 18, "humidity": 85, "wind": 5.0, "precip": 5.0},
        # 눈 오는 날씨
        {"temp": -2, "humidity": 80, "wind": 6.0, "precip": 3.0},
        # 바람 많은 날씨
        {"temp": 25, "humidity": 40, "wind": 12.0, "precip": 0.0},
    ]

    # 각 시나리오에 대해 여러 시간대의 관측값 생성
    for i, scenario in enumerate(scenarios):
        for hour_offset in range(0, 72, 3):  # 3시간 간격으로 72시간 (3일)
            observations = []

            for time_offset in range(3):  # 3시간 윈도우
                timestamp = base_time - timedelta(hours=hour_offset + time_offset, days=i)

                # 약간의 노이즈 추가
                import random
                temp_noise = random.uniform(-2, 2)
                humidity_noise = random.uniform(-5, 5)
                wind_noise = random.uniform(-1, 1)

                obs = WeatherObservation(
                    timestamp=timestamp,
                    temperature_c=scenario["temp"] + temp_noise,
                    wind_speed_ms=max(0, scenario["wind"] + wind_noise),
                    precipitation_mm=scenario["precip"],
                    relative_humidity=max(0, min(100, scenario["humidity"] + humidity_noise)),
                    precipitation_type="rain" if scenario["precip"] > 0 and scenario["temp"] > 2 else
                                    "snow" if scenario["precip"] > 0 else "none"
                )
                observations.append(obs)

            observations_sets.append(observations)

    logger.info(f"Created {len(observations_sets)} observation sets")
    return observations_sets


def generate_comfort_scores(observations_sets: List[List[WeatherObservation]]) -> List[float]:
    """편안함 점수 생성 (간단한 휴리스틱 기반)"""

    logger.info("Generating comfort scores")

    comfort_scores = []

    for observations in observations_sets:
        latest_obs = observations[-1]

        # 간단한 편안함 점수 계산
        temp = latest_obs.temperature_c
        humidity = latest_obs.relative_humidity or 50
        wind = latest_obs.wind_speed_ms
        precip = latest_obs.precipitation_mm

        # 온도 점수 (18-24°C 최적)
        if 18 <= temp <= 24:
            temp_score = 1.0
        elif 10 <= temp <= 30:
            temp_score = 0.7
        else:
            temp_score = 0.3

        # 습도 점수 (40-60% 최적)
        if 40 <= humidity <= 60:
            humidity_score = 1.0
        elif 30 <= humidity <= 70:
            humidity_score = 0.7
        else:
            humidity_score = 0.4

        # 바람 점수 (0-5m/s 양호)
        if wind <= 5:
            wind_score = 1.0
        elif wind <= 10:
            wind_score = 0.6
        else:
            wind_score = 0.2

        # 강수 점수
        if precip == 0:
            precip_score = 1.0
        elif precip <= 2:
            precip_score = 0.7
        elif precip <= 5:
            precip_score = 0.4
        else:
            precip_score = 0.1

        # 종합 점수 (가중 평균)
        comfort_score = (
            temp_score * 0.3 +
            humidity_score * 0.2 +
            wind_score * 0.2 +
            precip_score * 0.3
        )

        comfort_scores.append(comfort_score)

    logger.info(f"Generated {len(comfort_scores)} comfort scores")
    logger.info(f"Comfort score range: {min(comfort_scores):.3f} - {max(comfort_scores):.3f}")

    return comfort_scores


def main():
    """메인 실행 함수"""

    logger.info("🚀 Starting Enhanced Model Training")

    try:
        # 프로젝트 경로 설정
        project_root = Path(__file__).resolve().parents[1]
        paths = ProjectPaths.from_root(project_root)

        # MLflow 설정
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow_manager = MLflowManager(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="enhanced-commute-training-demo"
        )

        # S3 설정 (선택적)
        s3_bucket = os.getenv("COMMUTE_S3_BUCKET")
        s3_manager = None
        if s3_bucket:
            s3_manager = S3StorageManager(paths=paths, bucket=s3_bucket)

        # 훈련 설정
        config = EnhancedTrainingConfig(
            min_training_samples=50,  # 테스트용으로 낮춤
            test_size=0.2,
            enable_feature_selection=True,
            enable_hyperparameter_tuning=True,
            use_time_series_split=True,
            n_splits=3,  # 테스트용으로 낮춤
            max_features=30
        )

        # 예측기 생성
        predictor = EnhancedCommutePredictor(
            config=config,
            mlflow_manager=mlflow_manager,
            s3_manager=s3_manager
        )

        # 샘플 데이터 생성
        logger.info("📊 Generating sample training data")
        observations_sets = create_sample_observations()
        comfort_scores = generate_comfort_scores(observations_sets)

        # 타임스탬프 생성
        timestamps = [obs[-1].timestamp for obs in observations_sets]

        # 모델 훈련 및 MLflow 추적
        logger.info("🧠 Training models with MLflow tracking")
        run_ids = predictor.train_and_track(
            observations_list=observations_sets,
            comfort_scores=comfort_scores,
            timestamps=timestamps,
            experiment_name="enhanced-commute-training-demo"
        )

        # 결과 출력
        logger.info("✅ Training completed successfully!")
        logger.info(f"📈 Trained models: {len(run_ids)}")
        logger.info(f"🏆 Best model: {predictor.best_model_name}")
        logger.info(f"📊 Selected features: {len(predictor.selected_features)}")

        # MLflow 실행 ID 출력
        print("\n" + "="*60)
        print("🎯 MLflow Run IDs:")
        for model_name, run_id in run_ids.items():
            print(f"   {model_name}: {run_id}")

        print(f"\n📊 Feature Engineering Stats:")
        print(f"   Total features generated: {len(predictor.feature_names)}")
        print(f"   Selected features: {len(predictor.selected_features)}")
        print(f"   Training samples: {len(observations_sets)}")

        print(f"\n🔍 Top Selected Features:")
        for i, feature in enumerate(predictor.selected_features[:10]):
            print(f"   {i+1}. {feature}")

        print(f"\n📍 MLflow Tracking URI: {mlflow_tracking_uri}")
        print("💡 View results in MLflow UI: mlflow ui")

        # 예측 테스트
        if observations_sets:
            logger.info("🔮 Testing prediction on sample data")

            test_obs = observations_sets[0]
            prediction_result = predictor.predict_enhanced(test_obs)

            print(f"\n🔮 Sample Prediction:")
            print(f"   Predicted comfort score: {prediction_result['prediction']:.3f}")
            print(f"   Confidence: {prediction_result['confidence']:.3f}")
            print(f"   Model used: {prediction_result['model_used']}")
            print(f"   Feature quality: {prediction_result['feature_quality']:.3f}")

        print("\n🎉 Enhanced training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()