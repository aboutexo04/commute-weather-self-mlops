# 통합 MLOps 시스템 가이드

## 🎯 개요

이 프로젝트는 기상청 ASOS 데이터를 활용한 출퇴근 날씨 쾌적도 예측을 위한 완전한 MLOps 시스템입니다. 실시간 데이터 수집부터 모델 배포, 모니터링까지 전체 ML 라이프사이클을 자동화합니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성요소

1. **데이터 파이프라인**: 실시간 ASOS 데이터 수집 및 품질 검증
2. **피처 엔지니어링**: 지능형 피처 선택 및 생성
3. **모델 훈련**: MLflow 기반 실험 추적 및 모델 관리
4. **모델 배포**: 단계별 배포 전략 (Staging → Canary → Production)
5. **모니터링**: 시스템 성능 및 모델 품질 모니터링
6. **로깅**: 구조화된 로그 수집 및 분석

### 기술 스택

- **웹 프레임워크**: FastAPI
- **워크플로우 오케스트레이션**: Prefect
- **실험 추적**: MLflow + WandB
- **스토리지**: AWS S3 + 로컬 파일시스템
- **모니터링**: psutil + 커스텀 메트릭 시스템
- **배포**: Docker (선택사항)

## 📁 프로젝트 구조

```
commute-weather-self-mlops/
├── src/commute_weather/
│   ├── data_sources/           # 데이터 소스 관리
│   │   ├── asos_collector.py   # ASOS 데이터 수집기
│   │   └── kma_api.py         # 기상청 API 인터페이스
│   ├── features/              # 피처 엔지니어링
│   │   ├── feature_engineering.py
│   │   └── feature_selector.py
│   ├── pipelines/             # ML 파이프라인
│   │   ├── realtime_mlops_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── evaluation_pipeline.py
│   ├── deployment/            # 모델 배포
│   │   └── model_deployer.py
│   ├── storage/              # 데이터 저장소 관리
│   │   └── s3_manager.py
│   ├── tracking/             # 실험 추적
│   │   └── wandb_manager.py
│   ├── evaluation/           # 모델 평가
│   │   └── model_evaluator.py
│   └── monitoring/           # 모니터링 & 로깅
│       ├── system_monitor.py
│       └── logging_manager.py
├── scripts/                  # 유틸리티 스크립트
│   ├── setup_wandb.py       # WandB 설정
│   ├── run_realtime_mlops.py # 파이프라인 실행
│   └── deploy_mlops.py      # 배포 스크립트
├── app.py                   # 기본 FastAPI 앱
├── app_mlops.py            # MLOps 기능 통합 앱
└── docs/                   # 문서
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n mlops python=3.10
conda activate mlops

# 의존성 설치
pip install -r requirements.txt
# 또는
pip install fastapi uvicorn requests pandas scikit-learn mlflow wandb boto3 s3fs prefect python-dotenv psutil
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 다음 내용을 설정:

```env
# 필수 설정
KMA_AUTH_KEY=your_kma_api_key_here
WANDB_API_KEY=your_wandb_api_key_here

# 선택적 설정
COMMUTE_S3_BUCKET=your-s3-bucket-name
MLFLOW_TRACKING_URI=file:///path/to/mlflow
WANDB_PROJECT=commute-weather-mlops
AWS_DEFAULT_REGION=ap-northeast-2
KMA_STATION_ID=108
```

### 3. 시스템 배포

```bash
# 전체 시스템 설정 및 배포
python scripts/deploy_mlops.py deploy

# 또는 단계별 실행
python scripts/deploy_mlops.py setup   # 환경 설정
python scripts/deploy_mlops.py test    # 파이프라인 테스트
python scripts/deploy_mlops.py run     # 애플리케이션 실행
```

### 4. 웹 애플리케이션 접속

- **메인 애플리케이션**: http://localhost:8000
- **MLOps 대시보드**: http://localhost:8000/mlops/health
- **MLflow UI**: http://localhost:5000

## 📊 주요 기능

### 1. 실시간 데이터 파이프라인

```python
# 수동 파이프라인 실행
from commute_weather.pipelines.realtime_mlops_pipeline import realtime_mlops_flow

result = realtime_mlops_flow(
    auth_key="your_api_key",
    station_id="108",
    use_s3=True,
    use_wandb=True
)
```

### 2. 모델 배포 관리

```python
# 새 모델 배포
POST /mlops/models/deploy
{
    "model_name": "weather_predictor",
    "model_version": "v1.2.0",
    "mlflow_run_id": "abc123",
    "deployment_strategy": "canary"
}

# 현재 모델 정보 조회
GET /mlops/models/current

# 모델 롤백
POST /mlops/models/rollback
```

### 3. 시스템 모니터링

```python
# 시스템 상태 확인
GET /mlops/health

# 메트릭 조회
GET /mlops/metrics

# 파이프라인 트리거
GET /mlops/pipeline/trigger
```

## 🔧 설정 및 커스터마이징

### WandB 설정

```bash
# WandB 설정 도우미 실행
python scripts/setup_wandb.py
```

### S3 스토리지 구조

```
your-s3-bucket/
├── raw_data/
│   ├── asos/
│   └── kma/
├── features/
│   ├── engineered/
│   └── selected/
├── models/
│   ├── staging/
│   ├── production/
│   └── archive/
├── predictions/
├── evaluation/
└── monitoring/
    └── logs/
```

### 모니터링 알림 규칙

```python
from commute_weather.monitoring import AlertRule

# 커스텀 알림 규칙 추가
custom_rule = AlertRule(
    name="high_prediction_latency",
    metric="prediction_latency",
    threshold=3000.0,  # 3초
    operator=">=",
    duration_minutes=5,
    severity="WARNING",
    message_template="예측 응답시간이 {value:.0f}ms로 높습니다."
)

monitor.add_alert_rule(custom_rule)
```

## 📈 성능 최적화

### 1. 데이터 수집 최적화

- **배치 처리**: 여러 관측소 데이터 동시 수집
- **캐싱**: 최근 데이터 메모리 캐싱
- **재시도 로직**: 네트워크 오류 자동 복구

### 2. 피처 엔지니어링 최적화

- **적응형 선택**: 데이터 품질에 따른 피처 선택
- **병렬 처리**: 다중 피처 동시 생성
- **메모리 효율성**: 청크 단위 처리

### 3. 모델 서빙 최적화

- **모델 캐싱**: 메모리 내 모델 로딩
- **배치 예측**: 여러 요청 일괄 처리
- **연결 풀링**: 데이터베이스/API 연결 재사용

## 🔒 보안 고려사항

### 1. API 키 관리

- `.env` 파일을 통한 환경변수 관리
- 프로덕션에서는 AWS Secrets Manager 사용 권장

### 2. 데이터 보안

- S3 버킷 암호화 설정
- IAM 역할 기반 접근 제어
- VPC 내부 통신 사용

### 3. 애플리케이션 보안

- HTTPS 사용 (프로덕션)
- Rate limiting 구현
- 입력 데이터 검증

## 🐛 문제 해결

### 일반적인 문제들

1. **WandB 로그인 실패**
   ```bash
   # API 키 재설정
   wandb login --relogin
   ```

2. **S3 연결 오류**
   ```bash
   # AWS 자격증명 확인
   aws configure list
   ```

3. **MLflow 서버 연결 실패**
   ```bash
   # MLflow 서버 재시작
   mlflow ui --backend-store-uri ./mlflow
   ```

4. **메모리 부족**
   - 피처 엔지니어링 배치 크기 조정
   - 메트릭 히스토리 크기 제한

### 로그 확인

```bash
# 애플리케이션 로그
tail -f logs/mlops.log

# 에러 로그
tail -f logs/error.log

# 실시간 모니터링
python -c "
from commute_weather.monitoring import get_global_system_monitor
monitor = get_global_system_monitor()
print(monitor.get_metrics_summary())
"
```

## 📚 API 문서

### 기본 예측 API

- `GET /predict/now` - 현재 날씨 정보
- `GET /predict/morning` - 출근길 예측
- `GET /predict/evening` - 퇴근길 예측

### MLOps 관리 API

- `GET /mlops/health` - 시스템 상태
- `GET /mlops/models/current` - 현재 모델 정보
- `POST /mlops/models/deploy` - 모델 배포
- `POST /mlops/models/rollback` - 모델 롤백
- `GET /mlops/pipeline/trigger` - 파이프라인 실행
- `GET /mlops/metrics` - 시스템 메트릭

### 헬스체크 및 유틸리티

- `GET /health` - 애플리케이션 상태
- `GET /api/test` - KMA API 연결 테스트

## 🔄 지속적 개선

### 1. 모델 재훈련

- 일일/주간 성능 평가
- 데이터 드리프트 감지
- 자동 재훈련 트리거

### 2. 시스템 최적화

- 성능 메트릭 모니터링
- 리소스 사용량 최적화
- 알림 규칙 조정

### 3. 기능 확장

- 새로운 데이터 소스 추가
- 고급 피처 엔지니어링
- 다중 모델 앙상블

## 🤝 기여 가이드

1. **이슈 리포팅**: GitHub Issues 사용
2. **코드 기여**: Pull Request 프로세스
3. **문서 개선**: 마크다운 형식 유지
4. **테스트**: 새 기능에 대한 테스트 추가

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🆘 지원

- **문서**: `docs/` 디렉토리 참조
- **예제**: `examples/` 디렉토리 확인
- **이슈**: GitHub Issues 페이지
- **토론**: GitHub Discussions

---

이 MLOps 시스템은 실시간 날씨 예측을 위한 완전한 솔루션을 제공합니다. 시스템의 각 구성요소는 독립적으로 사용할 수 있도록 설계되어 있어 필요에 따라 부분적으로 도입하거나 확장할 수 있습니다.