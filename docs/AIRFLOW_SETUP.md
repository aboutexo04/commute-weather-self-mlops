# Apache Airflow 설정 가이드

## 개요

Apache Airflow를 사용하여 MLOps 파이프라인의 데이터 수집, 모델 훈련, 시스템 모니터링을 자동화합니다.

## 아키텍처

### Database Architecture
- **PostgreSQL**: Airflow 메타데이터 전용 (DAG 실행 히스토리, 작업 상태, 로그)
- **S3**: 실제 데이터 저장소 (날씨 데이터, 모델 아티팩트, 로그 파일)

### 서비스 구성
- **airflow-webserver**: Airflow UI (포트 8080)
- **airflow-scheduler**: DAG 스케줄링 및 작업 실행
- **airflow-postgres**: Airflow 메타데이터 데이터베이스

## DAG 구성

### 1. 데이터 수집 DAG (`data_collection_dag.py`)
- **스케줄**: 매시간 정각 (`0 * * * *`)
- **목적**: KMA API에서 날씨 데이터 수집하여 S3에 저장
- **작업 흐름**:
  ```
  collect_weather_data → validate_data_quality → update_data_catalog → health_check
  ```

### 2. 모델 훈련 DAG (`model_training_dag.py`)
- **스케줄**: 매일 새벽 2시 (`0 2 * * *`)
- **목적**: 일별 모델 훈련 및 성능 평가
- **작업 흐름**:
  ```
  check_data_readiness → prepare_training_data → train_model → evaluate_model → deploy_model
  ```

### 3. 모니터링 DAG (`monitoring_dag.py`)
- **스케줄**: 30분마다 (`*/30 * * * *`)
- **목적**: 시스템 성능 모니터링 및 데이터 드리프트 감지
- **작업 흐름**:
  ```
  [system_health, data_quality, model_performance] → generate_report → send_alerts → cleanup
  ```

## 설정 파일

### airflow.cfg
```ini
[core]
# DAG 디렉토리
dags_folder = ./dags

# 실행자 설정 (프로덕션용)
executor = LocalExecutor

# PostgreSQL 데이터베이스 (프로덕션용)
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow

# 예제 DAG 로드 비활성화
load_examples = False

[webserver]
# 웹서버 포트
web_server_port = 8080

# 기본 UI 타임존
default_ui_timezone = Asia/Seoul

# 인증 비활성화 (개발용)
authenticate = False
```

### Docker Compose 환경변수
```yaml
environment:
  - AIRFLOW__CORE__EXECUTOR=LocalExecutor
  - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@airflow-postgres/airflow
  - AIRFLOW__CORE__DAGS_FOLDER=/app/dags
  - AIRFLOW__CORE__LOAD_EXAMPLES=False
  - AIRFLOW__WEBSERVER__DEFAULT_UI_TIMEZONE=Asia/Seoul
  - AIRFLOW__CORE__FERNET_KEY=fernet_key_here_32_character_string
  - PYTHONPATH=/app/src
```

## 배포 방법

### 1. 로컬 개발 환경
```bash
# SQLite 사용 (간단한 테스트용)
export AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:///./airflow/airflow.db
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@admin.com
airflow webserver &
airflow scheduler
```

### 2. Docker 프로덕션 환경
```bash
# 전체 서비스 시작
docker-compose -f docker-compose.production.yml up -d

# Airflow 서비스만 시작
docker-compose -f docker-compose.production.yml up -d airflow-postgres airflow-webserver airflow-scheduler
```

## 모니터링

### Airflow UI 접근
- **URL**: http://localhost:8080 (로컬) 또는 http://EC2-IP:8080
- **사용자**: admin / admin

### 주요 모니터링 항목
1. **DAG 실행 상태**: 각 DAG의 성공/실패 여부
2. **작업 지연**: 예상 실행 시간 대비 지연 발생
3. **데이터 품질**: 수집된 데이터의 품질 점수
4. **시스템 리소스**: CPU, 메모리, 디스크 사용량

## 문제 해결

### 일반적인 문제
1. **PostgreSQL 연결 실패**
   ```bash
   # 컨테이너 상태 확인
   docker ps | grep postgres

   # 로그 확인
   docker logs mlops-airflow-postgres
   ```

2. **DAG 인식 안됨**
   ```bash
   # DAG 폴더 마운트 확인
   docker exec mlops-airflow-webserver ls -la /app/dags

   # Python 경로 확인
   docker exec mlops-airflow-webserver echo $PYTHONPATH
   ```

3. **S3 액세스 실패**
   ```bash
   # AWS 자격증명 확인
   docker exec mlops-airflow-webserver env | grep AWS

   # S3 연결 테스트
   docker exec mlops-airflow-webserver aws s3 ls s3://my-mlops-symun/
   ```

### 로그 확인
```bash
# 웹서버 로그
docker logs mlops-airflow-webserver

# 스케줄러 로그
docker logs mlops-airflow-scheduler

# 특정 작업 로그 (Airflow UI에서 확인 가능)
```

## 성능 최적화

### 리소스 제한
```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

### 동시 실행 제한
- `parallelism = 32`: 전체 동시 실행 작업 수
- `dag_concurrency = 16`: DAG별 동시 실행 작업 수
- `max_active_runs_per_dag = 1`: DAG별 최대 활성 실행 수

## 보안 고려사항

1. **인증 활성화** (프로덕션 환경)
   ```ini
   [webserver]
   authenticate = True
   auth_backend = airflow.auth.backends.password_auth
   ```

2. **Fernet 키 설정**
   ```bash
   # Fernet 키 생성
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

3. **네트워크 보안**
   - Airflow UI는 내부 네트워크에서만 접근
   - PostgreSQL은 Airflow 서비스에서만 접근

## 백업 및 복구

### 메타데이터 백업
```bash
# PostgreSQL 백업
docker exec mlops-airflow-postgres pg_dump -U airflow airflow > airflow_backup.sql

# 복구
docker exec -i mlops-airflow-postgres psql -U airflow airflow < airflow_backup.sql
```

### DAG 백업
```bash
# DAG 파일은 Git으로 관리
git add dags/
git commit -m "Update DAGs"
git push
```