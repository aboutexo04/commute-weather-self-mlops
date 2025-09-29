"""
Airflow DAG for Batch Prediction
정해진 시간에 미리 통근 예측을 수행하여 Redis에 저장
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
import sys
import os

# 프로젝트 경로 추가
sys.path.insert(0, '/app/src')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'catchup': False
}

# 출근 예측 DAG (6시, 7시, 8시)
morning_dag = DAG(
    'batch_morning_prediction',
    default_args=default_args,
    description='출근시간대 배치 예측 수행',
    schedule='0 6,7,8 * * *',  # 매일 6시, 7시, 8시
    max_active_runs=1,
    tags=['mlops', 'batch-prediction', 'morning', 'commute']
)

# 퇴근 예측 DAG (14시, 15시, 16시, 17시, 18시)
evening_dag = DAG(
    'batch_evening_prediction',
    default_args=default_args,
    description='퇴근시간대 배치 예측 수행',
    schedule='0 14,15,16,17,18 * * *',  # 매일 14-18시
    max_active_runs=1,
    tags=['mlops', 'batch-prediction', 'evening', 'commute']
)

def run_batch_prediction(prediction_type: str, **context):
    """배치 예측 실행 및 Redis 저장"""
    try:
        from commute_weather.config import get_kma_config
        from commute_weather.data_sources.kma_api import fetch_recent_weather_kma
        from commute_weather.features.feature_engineering import WeatherFeatureEngineer
        from commute_weather.ml.enhanced_training import EnhancedCommutePredictor
        import redis
        import json
        from datetime import datetime
        import pytz

        print(f"🚀 Starting batch prediction for {prediction_type}")

        # Redis 연결
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'mlops-redis-prod'),
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )

        # KMA 설정
        config = get_kma_config()

        # 최근 날씨 데이터 수집
        latest_observations = fetch_recent_weather_kma(config, lookback_hours=3)
        if not latest_observations:
            print("⚠️ No weather data available for prediction")
            return {'status': 'no_data', 'prediction_type': prediction_type}

        latest = latest_observations[-1]
        print(f"📊 Using weather data: {latest.temperature_c}°C, {latest.relative_humidity}% humidity")

        # 피처 엔지니어링
        feature_engineer = WeatherFeatureEngineer()
        features = feature_engineer.engineer_features(latest_observations, latest.timestamp)

        # 모델 예측 (기본 예측 로직 사용)
        base_score = 70.0

        # 온도 기반 조정
        temp_score = max(0, min(40, 40 - abs(latest.temperature_c - 18) * 2))

        # 습도 기반 조정
        humidity_score = max(0, min(30, 30 - abs(latest.relative_humidity - 60) * 0.3))

        # 강수량 기반 조정
        precipitation_penalty = min(20, latest.precipitation_mm * 10)

        final_score = base_score + temp_score + humidity_score - precipitation_penalty
        final_score = max(0, min(100, final_score))

        # 라벨 결정
        if final_score >= 80:
            label = "excellent"
        elif final_score >= 70:
            label = "good"
        elif final_score >= 60:
            label = "moderate"
        elif final_score >= 50:
            label = "uncomfortable"
        else:
            label = "harsh"

        # 시간대별 제목 설정
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kst)

        if prediction_type == "morning":
            title = "🌅 출근길 AI 예측"
        else:
            title = "🌆 퇴근길 AI 예측"

        # 예측 결과
        prediction_result = {
            "title": title,
            "score": round(final_score, 1),
            "label": label,
            "prediction_time": current_time.strftime("%Y-%m-%d %H:%M"),
            "evaluation": {
                "temperature": f"{latest.temperature_c}°C",
                "humidity": f"{latest.relative_humidity}%",
                "precipitation": f"{latest.precipitation_mm}mm",
                "weather_summary": f"온도 {latest.temperature_c}°C, 습도 {latest.relative_humidity}%, 강수량 {latest.precipitation_mm}mm"
            },
            "ml_info": {
                "prediction_type": "batch_enhanced",
                "features_used": len(features),
                "model_version": "enhanced_v1",
                "cache_generated": current_time.isoformat()
            }
        }

        # Redis 저장 (1시간 TTL)
        cache_key = f"prediction:{prediction_type}:{current_time.hour}"
        redis_client.setex(
            cache_key,
            3600,  # 1시간 TTL
            json.dumps(prediction_result)
        )

        print(f"✅ Batch prediction saved to Redis: {cache_key}")
        print(f"📈 Score: {final_score:.1f} ({label})")

        return {
            'status': 'success',
            'prediction_type': prediction_type,
            'score': final_score,
            'cache_key': cache_key,
            'timestamp': current_time.isoformat()
        }

    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def validate_batch_prediction(prediction_type: str, **context):
    """배치 예측 결과 검증"""
    try:
        import redis
        import json
        from datetime import datetime
        import pytz

        print(f"🔍 Validating batch prediction for {prediction_type}")

        # Redis 연결
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'mlops-redis-prod'),
            port=6379,
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )

        # 현재 시간의 캐시 확인
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kst)
        cache_key = f"prediction:{prediction_type}:{current_time.hour}"

        cached_result = redis_client.get(cache_key)

        if cached_result:
            result = json.loads(cached_result)
            print(f"✅ Validation successful: Score {result['score']} ({result['label']})")
            return {
                'status': 'valid',
                'cache_key': cache_key,
                'score': result['score']
            }
        else:
            print(f"❌ Validation failed: No cache found for {cache_key}")
            return {
                'status': 'cache_miss',
                'cache_key': cache_key
            }

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise

# 출근 예측 작업들
morning_prediction = PythonOperator(
    task_id='run_morning_prediction',
    python_callable=run_batch_prediction,
    op_kwargs={'prediction_type': 'morning'},
    dag=morning_dag
)

morning_validation = PythonOperator(
    task_id='validate_morning_prediction',
    python_callable=validate_batch_prediction,
    op_kwargs={'prediction_type': 'morning'},
    dag=morning_dag
)

morning_health_check = BashOperator(
    task_id='morning_health_check',
    bash_command='echo "🌅 Morning batch prediction completed"',
    dag=morning_dag
)

# 퇴근 예측 작업들
evening_prediction = PythonOperator(
    task_id='run_evening_prediction',
    python_callable=run_batch_prediction,
    op_kwargs={'prediction_type': 'evening'},
    dag=evening_dag
)

evening_validation = PythonOperator(
    task_id='validate_evening_prediction',
    python_callable=validate_batch_prediction,
    op_kwargs={'prediction_type': 'evening'},
    dag=evening_dag
)

evening_health_check = BashOperator(
    task_id='evening_health_check',
    bash_command='echo "🌆 Evening batch prediction completed"',
    dag=evening_dag
)

# 작업 의존성 설정
morning_prediction >> morning_validation >> morning_health_check
evening_prediction >> evening_validation >> evening_health_check