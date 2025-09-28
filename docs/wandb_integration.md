# WandB Integration Guide

WandB(Weights & Biases)가 commute-weather MLOps 시스템에 통합되어 실험 추적, 모델 성능 모니터링, 실시간 알림 등을 제공합니다.

## 설정

### 1. WandB 계정 및 API 키 설정

```bash
# WandB 로그인
wandb login

# 또는 환경변수로 API 키 설정
export WANDB_API_KEY="your_api_key_here"
```

### 2. 프로젝트 설정

기본적으로 `commute-weather-mlops` 프로젝트를 사용하지만, 사용자 정의 가능합니다:

```bash
# 기본 프로젝트 사용
python scripts/run_realtime_mlops.py realtime

# 사용자 정의 프로젝트
python scripts/run_realtime_mlops.py realtime --wandb-project "my-weather-project"

# 팀/조직 프로젝트
python scripts/run_realtime_mlops.py realtime --wandb-entity "my-team" --wandb-project "weather-mlops"
```

## 추적되는 메트릭

### 실시간 파이프라인

- **데이터 품질**: 수집된 기상 데이터의 품질 점수
- **피처 엔지니어링**: 생성된 피처 수, 품질, 신뢰도
- **모델 성능**: 예측값, 신뢰도, 모델 정보
- **파이프라인 성능**: 실행 시간, 효율성, 성공률

### 모델 평가

- **성능 메트릭**: R², MAE, RMSE, MAPE
- **품질 지표**: 데이터 품질, 피처 품질, 예측 신뢰도
- **검증 결과**: 모델 유효성, 재훈련 필요성
- **운영 메트릭**: 지연 시간, 모델 크기

### 추가 기능

- **예측 분포**: 예측값의 히스토그램 및 분포
- **임계값 비교**: 현재 성능 vs 임계값 비교
- **알림**: 모델 성능 저하, 재훈련 필요시 자동 알림

## 사용 예시

### 1. 실시간 파이프라인 실행

```bash
# WandB 활성화
python scripts/run_realtime_mlops.py realtime --use-wandb

# WandB 비활성화
python scripts/run_realtime_mlops.py realtime --no-use-wandb
```

### 2. 모델 평가 실행

```bash
# 24시간 데이터로 모델 평가
python scripts/run_realtime_mlops.py evaluation --hours-back 24

# 사용자 정의 WandB 설정으로 평가
python scripts/run_realtime_mlops.py evaluation \
    --hours-back 12 \
    --wandb-project "model-evaluation" \
    --wandb-entity "my-team"
```

### 3. 연속 모니터링

```bash
# 1시간마다 실시간 파이프라인 + WandB 추적
python scripts/run_realtime_mlops.py continuous \
    --interval 60 \
    --use-wandb \
    --wandb-project "production-monitoring"
```

## WandB 대시보드 활용

### 1. 실시간 모니터링

- **Metrics**: 실시간 성능 지표 모니터링
- **Runs**: 각 파이프라인 실행 기록 조회
- **Alerts**: 성능 저하 또는 장애 알림

### 2. 모델 성능 분석

- **Performance Trends**: 시간에 따른 모델 성능 변화
- **Prediction Distribution**: 예측값 분포 및 패턴 분석
- **Validation Results**: 모델 검증 결과 추적

### 3. 비교 분석

- **Model Comparison**: 다른 모델 버전 간 성능 비교
- **A/B Testing**: 실험적 모델과 프로덕션 모델 비교
- **Historical Analysis**: 과거 성능 데이터와 비교

## 고급 설정

### 1. Sweep을 통한 하이퍼파라미터 최적화

```python
from commute_weather.tracking import WandBManager

# Sweep 설정
sweep_config = {
    'method': 'random',
    'metric': {'name': 'r2_score', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.01, 0.1, 0.2]},
        'n_estimators': {'values': [50, 100, 200]}
    }
}

sweep_id = WandBManager.setup_sweep(
    sweep_config,
    project_name="hyperparameter-tuning"
)
```

### 2. 오프라인 모드

인터넷 연결이 제한된 환경에서 사용:

```bash
# 오프라인 모드 설정
export WANDB_MODE=offline

# 나중에 동기화
wandb sync
```

### 3. 사용자 정의 메트릭

```python
from commute_weather.tracking import get_global_wandb_manager

wandb_manager = get_global_wandb_manager()

# 사용자 정의 메트릭 로깅
wandb_manager.log_metrics({
    "custom_weather_score": 85.2,
    "data_freshness_minutes": 5.0,
    "prediction_accuracy_24h": 0.92
})
```

## 문제 해결

### 1. 로그인 문제

```bash
# API 키 재설정
wandb login --relogin

# 수동 API 키 설정
export WANDB_API_KEY="your_key"
```

### 2. 프로젝트 접근 권한

- WandB 프로젝트가 존재하는지 확인
- 팀/조직 권한 확인
- 올바른 entity 이름 사용

### 3. 성능 최적화

```bash
# 로깅 빈도 조절
export WANDB_LOG_LEVEL=WARNING

# 동기화 간격 조절 (초)
export WANDB_RESUME_WAIT=5
```

## 보안 고려사항

1. **API 키 보안**: 환경변수 또는 비밀 관리 시스템 사용
2. **데이터 프라이버시**: 민감한 데이터 로깅 방지
3. **프로젝트 권한**: 적절한 팀/개인 권한 설정

## 참고 자료

- [WandB 공식 문서](https://docs.wandb.ai/)
- [MLOps 모범 사례](https://wandb.ai/site/articles/mlops-best-practices)
- [실험 추적 가이드](https://docs.wandb.ai/guides/track)