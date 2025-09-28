#!/bin/bash

# MLOps 시스템 배포 스크립트
# Usage: ./scripts/deploy.sh [dev|prod]

set -e

ENVIRONMENT=${1:-dev}
PROJECT_NAME="commute-weather-mlops"

echo "🚀 Starting MLOps deployment for $ENVIRONMENT environment..."

# 환경 변수 파일 확인
if [ "$ENVIRONMENT" = "prod" ]; then
    ENV_FILE=".env.production"
    COMPOSE_FILE="docker-compose.production.yml"
else
    ENV_FILE=".env"
    COMPOSE_FILE="docker-compose.yml"
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file $ENV_FILE not found!"
    echo "Please create it from the template:"
    if [ "$ENVIRONMENT" = "prod" ]; then
        echo "cp .env.production .env"
    else
        echo "cp .env.example .env"
    fi
    exit 1
fi

echo "✅ Using environment file: $ENV_FILE"
echo "✅ Using compose file: $COMPOSE_FILE"

# Docker와 Docker Compose 확인
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed!"
    exit 1
fi

# 기존 컨테이너 정리 (선택사항)
read -p "🤔 Stop and remove existing containers? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 Stopping existing containers..."
    docker-compose -f $COMPOSE_FILE down --remove-orphans

    echo "🧹 Cleaning up unused Docker resources..."
    docker system prune -f
fi

# 필요한 디렉토리 생성
echo "📁 Creating necessary directories..."
mkdir -p logs data ssl monitoring/grafana/dashboards monitoring/grafana/datasources

# 이미지 빌드
echo "🔨 Building Docker images..."
if [ "$ENVIRONMENT" = "prod" ]; then
    docker-compose -f $COMPOSE_FILE build --no-cache
else
    docker-compose -f $COMPOSE_FILE build
fi

# 서비스 시작
echo "🚀 Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# 서비스 상태 확인
echo "⏳ Waiting for services to start..."
sleep 30

echo "🔍 Checking service health..."
docker-compose -f $COMPOSE_FILE ps

# 헬스 체크
echo "🏥 Running health checks..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt/$max_attempts: Checking main application..."

    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Main application is healthy!"
        break
    else
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ Health check failed after $max_attempts attempts"
            echo "📋 Container logs:"
            docker-compose -f $COMPOSE_FILE logs --tail=50 mlops-app
            exit 1
        fi
        echo "⏳ Waiting 10 seconds before retry..."
        sleep 10
        ((attempt++))
    fi
done

# MLflow 서비스 확인 (production에서만)
if [ "$ENVIRONMENT" = "prod" ]; then
    echo "🔍 Checking MLflow server..."
    if curl -f http://localhost:5001 > /dev/null 2>&1; then
        echo "✅ MLflow server is running!"
    else
        echo "⚠️ MLflow server might not be ready yet"
    fi
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
echo "   Main Application: http://localhost:8000"
if [ "$ENVIRONMENT" = "prod" ]; then
    echo "   MLflow Server:    http://localhost:5001"
    echo "   Grafana:          http://localhost:3000"
    echo "   Prometheus:       http://localhost:9090"
fi
echo ""
echo "📋 Useful commands:"
echo "   View logs:        docker-compose -f $COMPOSE_FILE logs -f"
echo "   Stop services:    docker-compose -f $COMPOSE_FILE down"
echo "   Restart:          docker-compose -f $COMPOSE_FILE restart"
echo "   Check status:     docker-compose -f $COMPOSE_FILE ps"
echo ""

if [ "$ENVIRONMENT" = "prod" ]; then
    echo "🔒 Production deployment notes:"
    echo "   - Make sure to configure SSL certificates in ./ssl/"
    echo "   - Update nginx.conf with your actual domain"
    echo "   - Set up proper firewall rules"
    echo "   - Configure log rotation"
    echo "   - Set up monitoring alerts"
fi

echo "🎯 Happy MLOps! Your weather prediction system is now running!"