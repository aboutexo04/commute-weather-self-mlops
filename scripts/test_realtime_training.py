#!/usr/bin/env python3
"""실시간 데이터 기반 훈련 시스템 테스트."""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from commute_weather.data.training_data_manager import RealTimeTrainingDataManager
from commute_weather.ml.enhanced_training import EnhancedCommutePredictor, EnhancedTrainingConfig
from commute_weather.ml.mlflow_integration import MLflowManager
from commute_weather.storage.s3_manager import S3StorageManager
from commute_weather.config import ProjectPaths
from commute_weather.data_sources.weather_api import WeatherObservation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_collected_data(storage: S3StorageManager) -> None:
    """실시간 수집 데이터 모방 (테스트용)"""

    logger.info("Creating mock collected data for testing")

    # 지난 7일간의 모의 데이터 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    current_date = start_date
    while current_date <= end_date:
        try:
            # 하루 24시간의 모의 KMA 데이터 생성
            mock_kma_data = generate_mock_kma_data(current_date)

            # S3에 저장 (원시 데이터 형태로)
            date_str = current_date.strftime("%Y/%m/%d")
            for hour in range(0, 24, 3):  # 3시간마다
                file_timestamp = current_date.replace(hour=hour, minute=0, second=0)
                filename = f"{file_timestamp.strftime('%H%M')}.txt"

                s3_path = f"raw_data/kma/108/{date_str}/{filename}"

                # 해당 시간의 KMA 원시 데이터 생성
                hour_data = mock_kma_data[hour:hour+3]  # 3시간치 데이터
                kma_raw_content = generate_kma_raw_format(hour_data, file_timestamp)

                storage.write_text(s3_path, kma_raw_content)

            logger.info(f"Created mock data for {current_date.date()}")

        except Exception as e:
            logger.warning(f"Failed to create mock data for {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    logger.info("Mock data creation completed")


def generate_mock_kma_data(date: datetime) -> List[Dict]:
    """하루치 모의 KMA 데이터 생성"""

    import random
    import math

    mock_data = []

    # 계절적 패턴 (월별)
    month = date.month
    if month in [12, 1, 2]:  # 겨울
        base_temp = random.uniform(-5, 5)
        base_humidity = random.uniform(40, 70)
    elif month in [3, 4, 5]:  # 봄
        base_temp = random.uniform(10, 20)
        base_humidity = random.uniform(50, 80)
    elif month in [6, 7, 8]:  # 여름
        base_temp = random.uniform(25, 35)
        base_humidity = random.uniform(70, 90)
    else:  # 가을
        base_temp = random.uniform(15, 25)
        base_humidity = random.uniform(40, 70)

    # 24시간 데이터 생성
    for hour in range(24):
        # 일일 온도 변화 패턴
        hour_factor = math.sin((hour - 6) * math.pi / 12)  # 오후 3시 최고, 오전 6시 최저
        temperature = base_temp + hour_factor * 5 + random.uniform(-2, 2)

        # 습도는 온도와 반비례 경향
        humidity = base_humidity - hour_factor * 10 + random.uniform(-5, 5)
        humidity = max(20, min(95, humidity))

        # 바람과 강수량
        wind_speed = max(0, random.uniform(0, 8) + random.uniform(-2, 2))

        # 강수량 (10% 확률로 비)
        precipitation = 0.0
        if random.random() < 0.1:
            precipitation = random.uniform(0.1, 5.0)

        # 강수 타입
        precip_type = "none"
        if precipitation > 0:
            if temperature <= 2:
                precip_type = "snow"
            else:
                precip_type = "rain"

        timestamp = date.replace(hour=hour, minute=0, second=0, microsecond=0)

        mock_data.append({
            "timestamp": timestamp,
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "precipitation": round(precipitation, 1),
            "precip_type": precip_type
        })

    return mock_data


def generate_kma_raw_format(data: List[Dict], start_time: datetime) -> str:
    """KMA 원시 데이터 형식으로 변환"""

    lines = []

    # 헤더 추가
    lines.append("# KMA Weather Data")
    lines.append("# STN YYMMDDHHMI TA HM WS RN")

    for item in data:
        timestamp_str = item["timestamp"].strftime("%Y%m%d%H%M")
        line = f"108 {timestamp_str} {item['temperature']} {item['humidity']} {item['wind_speed']} {item['precipitation']}"
        lines.append(line)

    return "\n".join(lines)


def test_real_time_training_pipeline():
    """실시간 훈련 파이프라인 테스트"""

    logger.info("🧪 Testing real-time training pipeline")

    try:
        # 프로젝트 설정
        project_root = Path(__file__).resolve().parents[1]
        paths = ProjectPaths.from_root(project_root)

        # S3 매니저 설정 (테스트용 - 로컬 파일시스템 사용)
        s3_bucket = "test-bucket"  # 실제로는 로컬 경로 사용
        storage = S3StorageManager(paths=paths, bucket=s3_bucket)

        # 1. 모의 수집 데이터 생성
        logger.info("📁 Creating mock collected data")
        create_mock_collected_data(storage)

        # 2. 실시간 훈련 데이터 매니저 테스트
        logger.info("📊 Testing training data manager")
        data_manager = RealTimeTrainingDataManager(
            storage_manager=storage,
            min_quality_score=0.5,  # 테스트용으로 낮춤
            min_observations_per_set=2
        )

        # 훈련 데이터 수집
        observation_sets, comfort_scores, stats = data_manager.collect_training_data(
            days_back=7,
            station_id="108"
        )

        print(f"\n📊 Training Data Collection Results:")
        print(f"   Observation sets: {len(observation_sets)}")
        print(f"   Total observations: {stats.total_observations}")
        print(f"   Average quality: {stats.average_quality_score:.3f}")
        print(f"   Data completeness: {stats.data_completeness:.3f}")
        print(f"   Comfort score range: {stats.comfort_score_range[0]:.3f} - {stats.comfort_score_range[1]:.3f}")

        if len(observation_sets) < 10:
            logger.warning("Insufficient training data - generating more mock data")
            # 더 많은 모의 데이터 생성...
            return

        # 3. 향상된 모델 훈련 테스트
        logger.info("🧠 Testing enhanced model training")

        # MLflow 설정
        mlflow_manager = MLflowManager(
            tracking_uri="file:./test_mlruns",
            experiment_name="realtime-training-test"
        )

        # 훈련 설정
        config = EnhancedTrainingConfig(
            min_training_samples=10,  # 테스트용으로 낮춤
            test_size=0.3,
            enable_feature_selection=True,
            use_time_series_split=True,
            n_splits=3,
            max_features=20
        )

        # 예측기 생성 및 훈련
        predictor = EnhancedCommutePredictor(
            config=config,
            mlflow_manager=mlflow_manager,
            s3_manager=storage
        )

        # 타임스탬프 생성
        timestamps = [obs[-1].timestamp for obs in observation_sets]

        # 훈련 및 추적
        run_ids = predictor.train_and_track(
            observations_list=observation_sets,
            comfort_scores=comfort_scores,
            timestamps=timestamps,
            experiment_name="realtime-training-test"
        )

        print(f"\n🎯 Model Training Results:")
        print(f"   Models trained: {len(run_ids)}")
        print(f"   Best model: {predictor.best_model_name}")
        print(f"   Selected features: {len(predictor.selected_features)}")

        print(f"\n📈 MLflow Run IDs:")
        for model_name, run_id in run_ids.items():
            print(f"   {model_name}: {run_id}")

        # 4. 예측 테스트
        if observation_sets:
            logger.info("🔮 Testing prediction with real-time data")

            # 최근 관측 데이터로 예측
            recent_obs = data_manager.get_recent_observations_for_prediction(
                hours_back=3,
                station_id="108"
            )

            if recent_obs:
                prediction_result = predictor.predict_enhanced(recent_obs)

                print(f"\n🔮 Prediction Test:")
                print(f"   Input observations: {len(recent_obs)}")
                print(f"   Predicted comfort: {prediction_result['prediction']:.3f}")
                print(f"   Confidence: {prediction_result['confidence']:.3f}")
                print(f"   Model used: {prediction_result['model_used']}")
                print(f"   Feature quality: {prediction_result['feature_quality']:.3f}")

        # 5. 데이터셋 요약
        dataset_summary = data_manager.create_training_dataset_summary(
            observation_sets, comfort_scores, stats
        )

        print(f"\n📋 Dataset Summary:")
        print(f"   Collection method: {dataset_summary['data_source']['collection_method']}")
        print(f"   Processing version: {dataset_summary['data_source']['processing_version']}")
        print(f"   Duration: {dataset_summary['dataset_info']['date_range']['duration_days']} days")

        logger.info("✅ Real-time training pipeline test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Real-time training test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """메인 실행 함수"""

    print("🧪 Real-Time Training Pipeline Test")
    print("=" * 50)

    try:
        test_real_time_training_pipeline()
        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()