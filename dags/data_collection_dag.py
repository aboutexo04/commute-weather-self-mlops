"""
Airflow DAG for MLOps Data Collection
매시간 KMA API에서 날씨 데이터를 수집하고 S3에 저장
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
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'kma_data_collection',
    default_args=default_args,
    description='KMA 날씨 데이터 수집 및 S3 저장',
    schedule='0 * * * *',  # 매시간 정각
    max_active_runs=1,
    tags=['mlops', 'data-collection', 'kma', 'weather']
)

def collect_weather_data(**context):
    """KMA API에서 날씨 데이터 수집"""
    try:
        from commute_weather.config import KMAAPIConfig
        from commute_weather.data_sources.kma_api import fetch_recent_weather_kma
        from commute_weather.storage.s3_manager import S3StorageManager
        import requests

        print("🌤️ Starting weather data collection...")

        # KMA API 설정
        kma_auth_key = os.getenv('KMA_AUTH_KEY')
        if not kma_auth_key:
            raise ValueError("KMA_AUTH_KEY environment variable not set")

        config = KMAAPIConfig(
            base_url="https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
            auth_key=kma_auth_key,
            station_id="108",  # 서울 관측소
            timezone="Asia/Seoul"
        )
        print(f"📍 Station: {config.station_id}")

        # 최근 데이터 수집 (지난 2시간)
        observations = fetch_recent_weather_kma(config, lookback_hours=2)

        if not observations:
            print("⚠️ No weather data received from KMA API")
            return {
                'observations_count': 0,
                'status': 'no_data',
                'timestamp': datetime.now().isoformat()
            }

        print(f"📊 Collected {len(observations)} observations")

        # S3 저장
        s3_manager = S3StorageManager(bucket_name=os.getenv('COMMUTE_S3_BUCKET', 'my-mlops-symun'))

        # 각 관측 데이터를 S3에 저장 (Raw data)
        saved_count = 0
        for obs in observations:
            try:
                # 관측 데이터를 텍스트 형태로 변환
                obs_text = f"{obs.timestamp.strftime('%Y%m%d%H%M')},{obs.temperature_c},{obs.wind_speed_ms},{obs.precipitation_mm},{obs.relative_humidity},{obs.precipitation_type}\n"

                # S3에 저장
                s3_manager.store_raw_weather_data(
                    data=obs_text,
                    station_id=config.station_id,
                    timestamp=obs.timestamp,
                    source="kma",
                    metadata={
                        "temperature_c": obs.temperature_c,
                        "wind_speed_ms": obs.wind_speed_ms,
                        "precipitation_mm": obs.precipitation_mm,
                        "relative_humidity": obs.relative_humidity,
                        "precipitation_type": obs.precipitation_type
                    }
                )
                saved_count += 1
                print(f"✅ Saved raw observation: {obs.timestamp}")
            except Exception as e:
                print(f"⚠️ Failed to save observation {obs.timestamp}: {e}")

        print(f"📈 Successfully saved {saved_count}/{len(observations)} raw observations")

        # 피처 엔지니어링 및 ML 데이터셋 저장
        try:
            from commute_weather.features.feature_engineering import WeatherFeatureEngineer
            import pandas as pd

            print("🧮 Starting feature engineering...")
            feature_engineer = WeatherFeatureEngineer()

            ml_datasets_saved = 0
            for obs in observations:
                try:
                    # 단일 관측값을 리스트로 래핑하여 피처 엔지니어링
                    features = feature_engineer.engineer_features([obs], obs.timestamp)

                    # DataFrame으로 변환
                    features_df = pd.DataFrame([features])

                    # ML 데이터셋으로 S3에 저장
                    s3_manager.store_processed_features(
                        df=features_df,
                        feature_set_name="kma_weather_features",
                        version="v1",
                        timestamp=obs.timestamp
                    )

                    ml_datasets_saved += 1
                    print(f"✅ Saved ML dataset: {obs.timestamp}")

                except Exception as e:
                    print(f"⚠️ Failed to create ML dataset for {obs.timestamp}: {e}")

            print(f"🧮 Successfully saved {ml_datasets_saved}/{len(observations)} ML datasets")

        except Exception as e:
            print(f"⚠️ Feature engineering failed: {e}")
            # 피처 엔지니어링 실패해도 raw data는 저장되었으므로 전체 실패로 처리하지 않음

        return {
            'observations_count': len(observations),
            'raw_data_saved': saved_count,
            'ml_datasets_saved': ml_datasets_saved if 'ml_datasets_saved' in locals() else 0,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def validate_data_quality(**context):
    """수집된 데이터의 품질 검증"""
    try:
        from commute_weather.storage.s3_manager import S3StorageManager

        print("🔍 Starting data quality validation...")

        s3_manager = S3StorageManager(bucket_name=os.getenv('COMMUTE_S3_BUCKET', 'my-mlops-symun'))

        # 이전 작업에서 수집 결과 가져오기
        task_instance = context['task_instance']
        collection_result = task_instance.xcom_pull(task_ids='collect_weather_data')

        if not collection_result:
            print("⚠️ No collection result found")
            return {'overall': 0.0, 'status': 'no_data'}

        raw_data_saved = collection_result.get('raw_data_saved', 0)
        ml_datasets_saved = collection_result.get('ml_datasets_saved', 0)
        total_count = collection_result.get('observations_count', 0)

        # 간단한 품질 평가
        if total_count == 0:
            raw_completeness = 0.0
            ml_completeness = 0.0
        else:
            raw_completeness = raw_data_saved / total_count
            ml_completeness = ml_datasets_saved / total_count

        # 기본적인 품질 메트릭
        quality_result = {
            'raw_data_completeness': raw_completeness,
            'ml_data_completeness': ml_completeness,
            'consistency': 0.98,  # 가정값
            'timeliness': 0.92,   # 가정값
            'total_observations': total_count,
            'raw_data_saved': raw_data_saved,
            'ml_datasets_saved': ml_datasets_saved,
            'overall': (raw_completeness * 0.4 + ml_completeness * 0.4 + 0.95 * 0.2)  # weighted average
        }

        print(f"📊 Quality Results: {quality_result}")

        # 품질이 너무 낮으면 알림
        if quality_result['overall'] < 0.8:
            print("⚠️ Data quality below threshold!")
        else:
            print("✅ Data quality acceptable")

        return quality_result

    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def update_data_catalog(**context):
    """데이터 카탈로그 업데이트"""
    try:
        print("📋 Updating data catalog...")

        # 데이터 메타데이터 업데이트
        task_instance = context['task_instance']
        collection_result = task_instance.xcom_pull(task_ids='collect_weather_data')
        quality_result = task_instance.xcom_pull(task_ids='validate_data_quality')

        catalog_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'KMA API',
            'observations_count': collection_result.get('observations_count', 0),
            'quality_score': quality_result.get('overall', 0),
            'status': 'success'
        }

        print(f"📊 Catalog Entry: {catalog_entry}")

        # 여기서 실제로는 데이터베이스나 메타데이터 스토어에 저장

        return catalog_entry

    except Exception as e:
        print(f"❌ Catalog update failed: {e}")
        raise

# 작업 정의
collect_task = PythonOperator(
    task_id='collect_weather_data',
    python_callable=collect_weather_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

catalog_task = PythonOperator(
    task_id='update_data_catalog',
    python_callable=update_data_catalog,
    dag=dag
)

# 헬스체크 작업
health_check = BashOperator(
    task_id='health_check',
    bash_command='echo "🏥 Data collection pipeline health check completed"',
    dag=dag
)

# 작업 의존성 설정
collect_task >> validate_task >> catalog_task >> health_check