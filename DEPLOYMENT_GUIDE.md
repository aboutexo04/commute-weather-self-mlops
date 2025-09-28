# 🚀 MLOps 시스템 배포 가이드

이 가이드는 출퇴근 날씨 예측 MLOps 시스템을 Docker로 배포하는 방법을 설명합니다.

## 📋 사전 요구사항

### 필수 소프트웨어
- Docker (20.10+)
- Docker Compose (2.0+)
- Git

### 시스템 요구사항
- **개발 환경**: 최소 4GB RAM, 2 CPU 코어
- **프로덕션 환경**: 최소 8GB RAM, 4 CPU 코어
- 디스크 공간: 10GB+ (로그, 데이터, 모델 저장용)

### 외부 서비스
- AWS S3 버킷 (데이터 및 모델 저장)
- KMA API 키 (기상청 데이터 수집)
- WandB 계정 (선택사항, 실험 추적)

## 🔧 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/aboutexo04/commute-weather-self-mlops.git
cd commute-weather-self-mlops
```

### 2. 환경 변수 설정

#### 개발 환경
```bash
cp .env.production .env
```

#### 프로덕션 환경
```bash
cp .env.production .env
```

`.env` 파일을 편집하여 실제 값으로 변경:

```bash
# AWS 자격증명
AWS_ACCESS_KEY_ID=your_actual_access_key
AWS_SECRET_ACCESS_KEY=your_actual_secret_key

# S3 버킷
COMMUTE_S3_BUCKET=your-production-bucket-name

# KMA API
KMA_AUTH_KEY=your_kma_api_key

# 데이터베이스 비밀번호 (프로덕션만)
POSTGRES_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_secure_redis_password
GRAFANA_PASSWORD=your_grafana_admin_password
```

## 🚀 배포 방법

### 개발 환경 배포

기본적인 개발 환경 배포:

```bash
# 간단한 개발 환경
docker-compose up -d

# 또는 배포 스크립트 사용
./scripts/deploy.sh dev
```

서비스 접속:
- **메인 애플리케이션**: http://localhost:8000
- **MLflow UI**: http://localhost:5001

### 프로덕션 환경 배포

완전한 프로덕션 환경 (Nginx, PostgreSQL, 모니터링 포함):

```bash
./scripts/deploy.sh prod
```

서비스 접속:
- **메인 애플리케이션**: http://localhost (Nginx를 통해)
- **MLflow UI**: http://localhost:5001
- **Grafana 대시보드**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## 📊 서비스 구성

### 개발 환경 서비스
- `mlops-app`: 메인 FastAPI 애플리케이션
- `redis`: 캐싱 및 작업 큐
- `mlflow-server`: MLflow 추적 서버
- `data-collector`: 실시간 데이터 수집

### 프로덕션 환경 추가 서비스
- `nginx`: 리버스 프록시 및 로드 밸런서
- `postgres`: PostgreSQL 데이터베이스
- `training-scheduler`: 모델 훈련 스케줄러
- `prometheus`: 메트릭 수집
- `grafana`: 모니터링 대시보드

## 🔍 상태 확인 및 관리

### 서비스 상태 확인
```bash
# 모든 컨테이너 상태
docker-compose ps

# 특정 서비스 로그
docker-compose logs -f mlops-app

# 헬스 체크
curl http://localhost:8000/health
```

### 서비스 관리
```bash
# 서비스 재시작
docker-compose restart mlops-app

# 서비스 중지
docker-compose down

# 완전 정리 (볼륨 포함)
docker-compose down -v
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 포트 충돌
```bash
# 사용 중인 포트 확인
sudo lsof -i :8000

# 프로세스 종료
sudo kill -9 PID
```

#### 2. 환경 변수 오류
```bash
# 환경 변수 확인
docker-compose exec mlops-app env | grep KMA_AUTH_KEY
```

#### 3. S3 연결 오류
```bash
# AWS 자격증명 테스트
docker-compose exec mlops-app python -c "import boto3; print(boto3.client('s3').list_buckets())"
```

#### 4. 메모리 부족
```bash
# 메모리 사용량 확인
docker stats

# 불필요한 컨테이너 정리
docker system prune -a
```

### 로그 확인
```bash
# 모든 서비스 로그
docker-compose logs

# 특정 서비스만
docker-compose logs mlops-app

# 실시간 로그 추적
docker-compose logs -f --tail=100 mlops-app
```

## 📈 모니터링 및 유지보수

### Grafana 대시보드 설정

1. http://localhost:3000 접속
2. admin / ${GRAFANA_PASSWORD}로 로그인
3. 기본 대시보드들이 자동으로 프로비저닝됨

### 정기 유지보수

#### 일일 점검
```bash
# 헬스 체크
curl http://localhost:8000/health

# 디스크 사용량 확인
df -h

# 메모리 사용량 확인
free -h
```

#### 주간 점검
```bash
# 로그 로테이션
docker-compose exec mlops-app find /app/logs -name "*.log" -mtime +7 -delete

# Docker 이미지 정리
docker image prune -a

# 시스템 업데이트
docker-compose pull
docker-compose up -d
```

## 🔒 보안 고려사항

### 프로덕션 배포 시 추가 설정

1. **SSL 인증서 설정**
   ```bash
   # SSL 인증서를 ./ssl/ 디렉토리에 배치
   mkdir -p ssl
   # cert.pem, key.pem 파일 배치
   ```

2. **방화벽 설정**
   ```bash
   # 필요한 포트만 열기
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw enable
   ```

3. **환경 변수 보안**
   - 실제 배포 시 강력한 비밀번호 사용
   - AWS IAM 역할 사용 고려
   - 시크릿 관리 도구 사용 (AWS Secrets Manager 등)

## 🚀 배포 자동화

### GitHub Actions (CI/CD)

`.github/workflows/deploy.yml` 파일 생성:

```yaml
name: Deploy MLOps System

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to server
        run: |
          # SSH를 통한 서버 배포 스크립트
          ssh user@your-server "cd /path/to/project && git pull && ./scripts/deploy.sh prod"
```

## 📞 지원 및 문의

문제가 발생하거나 추가 도움이 필요한 경우:

1. **GitHub Issues**: 버그 리포트 및 기능 요청
2. **로그 수집**: 문제 발생 시 관련 로그 첨부
3. **환경 정보**: OS, Docker 버전, 하드웨어 스펙 제공

---

## 📚 추가 자료

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Grafana 문서](https://grafana.com/docs/)

이제 완전한 MLOps 시스템을 프로덕션 환경에서 실행할 수 있습니다! 🎉