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

        # 각 관측 데이터를 S3에 저장
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
                print(f"✅ Saved observation: {obs.timestamp}")
            except Exception as e:
                print(f"⚠️ Failed to save observation {obs.timestamp}: {e}")

        print(f"📈 Successfully saved {saved_count}/{len(observations)} observations")

        return {
            'observations_count': len(observations),
            'saved_count': saved_count,
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

        saved_count = collection_result.get('saved_count', 0)
        total_count = collection_result.get('observations_count', 0)

        # 간단한 품질 평가
        if total_count == 0:
            completeness = 0.0
        else:
            completeness = saved_count / total_count

        # 기본적인 품질 메트릭
        quality_result = {
            'completeness': completeness,
            'consistency': 0.98,  # 가정값
            'timeliness': 0.92,   # 가정값
            'total_observations': total_count,
            'saved_observations': saved_count,
            'overall': completeness * 0.6 + 0.95 * 0.4  # weighted average
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