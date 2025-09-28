"""모델 배포 및 버전 관리 시스템."""

from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd
import mlflow
from zoneinfo import ZoneInfo

from ..storage import S3StorageManager
from ..tracking import WandBManager
from ..evaluation import ModelEvaluator, ValidationResult

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """배포 단계 정의."""
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class DeploymentStrategy(Enum):
    """배포 전략 정의."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TEST = "a_b_test"
    IMMEDIATE = "immediate"


@dataclass
class ModelVersion:
    """모델 버전 정보."""
    model_name: str
    version: str
    mlflow_run_id: str
    stage: DeploymentStage
    performance_metrics: Dict[str, float]
    deployment_timestamp: datetime
    traffic_percentage: float = 0.0

    # 메타데이터
    metadata: Dict[str, Any] = None
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    last_health_check: Optional[datetime] = None


@dataclass
class DeploymentPlan:
    """배포 계획."""
    source_model: ModelVersion
    target_stage: DeploymentStage
    strategy: DeploymentStrategy

    # 전략별 설정
    canary_traffic_steps: List[float] = None  # [10, 25, 50, 100]
    canary_duration_minutes: int = 60
    rollback_threshold: float = 0.1  # 성능 저하 10% 이상시 롤백

    # 검증 설정
    validation_required: bool = True
    approval_required: bool = False
    auto_promotion: bool = False


class ModelDeployer:
    """모델 배포 관리자."""

    def __init__(
        self,
        s3_manager: S3StorageManager,
        model_evaluator: ModelEvaluator,
        mlflow_tracking_uri: Optional[str] = None,
        wandb_manager: Optional[WandBManager] = None
    ):
        self.s3_manager = s3_manager
        self.model_evaluator = model_evaluator
        self.wandb_manager = wandb_manager

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # 현재 배포된 모델들 추적
        self.deployed_models: Dict[DeploymentStage, List[ModelVersion]] = {
            DeploymentStage.STAGING: [],
            DeploymentStage.CANARY: [],
            DeploymentStage.PRODUCTION: [],
            DeploymentStage.ROLLBACK: []
        }

        # 배포 설정
        self.deployment_config = {
            "max_staging_models": 5,
            "max_canary_models": 2,
            "max_production_models": 1,
            "health_check_interval_minutes": 15,
            "performance_degradation_threshold": 0.1,
            "auto_rollback_enabled": True
        }

    def create_deployment_plan(
        self,
        model_name: str,
        model_version: str,
        mlflow_run_id: str,
        target_stage: DeploymentStage,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        **kwargs
    ) -> DeploymentPlan:
        """배포 계획 생성."""

        # 소스 모델 정보 수집
        try:
            run = mlflow.get_run(mlflow_run_id)
            performance_metrics = {
                "r2": float(run.data.metrics.get("r2", 0.0)),
                "mae": float(run.data.metrics.get("mae", 0.0)),
                "mape": float(run.data.metrics.get("mape", 0.0)),
                "prediction_confidence": float(run.data.metrics.get("prediction_confidence", 0.0))
            }

            source_model = ModelVersion(
                model_name=model_name,
                version=model_version,
                mlflow_run_id=mlflow_run_id,
                stage=DeploymentStage.STAGING,
                performance_metrics=performance_metrics,
                deployment_timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
                metadata=dict(run.data.params)
            )

        except Exception as e:
            logger.error(f"Failed to retrieve model info for {mlflow_run_id}: {e}")
            raise ValueError(f"Invalid MLflow run ID: {mlflow_run_id}")

        # 배포 계획 생성
        plan = DeploymentPlan(
            source_model=source_model,
            target_stage=target_stage,
            strategy=strategy,
            **kwargs
        )

        # 전략별 기본 설정
        if strategy == DeploymentStrategy.CANARY:
            plan.canary_traffic_steps = kwargs.get("canary_traffic_steps", [10, 25, 50, 100])
            plan.canary_duration_minutes = kwargs.get("canary_duration_minutes", 60)
        elif strategy == DeploymentStrategy.A_B_TEST:
            plan.canary_traffic_steps = kwargs.get("traffic_split", [50, 50])

        logger.info(f"Created deployment plan: {model_name} v{model_version} → {target_stage.value}")
        return plan

    def validate_deployment_readiness(
        self,
        deployment_plan: DeploymentPlan
    ) -> Tuple[bool, List[str]]:
        """배포 준비 상태 검증."""

        validation_issues = []
        model = deployment_plan.source_model

        # 1. 성능 임계값 검증
        r2_score = model.performance_metrics.get("r2", 0.0)
        if r2_score < 0.7:
            validation_issues.append(f"R² score ({r2_score:.3f}) below threshold (0.7)")

        mae = model.performance_metrics.get("mae", float('inf'))
        if mae > 0.5:
            validation_issues.append(f"MAE ({mae:.3f}) above threshold (0.5)")

        # 2. 모델 크기 및 복잡성 검증
        try:
            # MLflow에서 모델 크기 확인
            model_info = mlflow.models.get_model_info(f"runs:/{model.mlflow_run_id}/model")
            # 모델 크기나 복잡성 검증 로직 추가 가능
        except Exception as e:
            validation_issues.append(f"Failed to validate model artifacts: {e}")

        # 3. 의존성 및 호환성 검증
        # 피처 스키마, 버전 호환성 등 검증

        # 4. 현재 프로덕션 모델과 비교
        if deployment_plan.target_stage == DeploymentStage.PRODUCTION:
            current_production = self._get_current_production_model()
            if current_production:
                current_r2 = current_production.performance_metrics.get("r2", 0.0)
                if r2_score < current_r2 * 0.95:  # 5% 이상 성능 저하
                    validation_issues.append(
                        f"New model performance ({r2_score:.3f}) significantly lower than "
                        f"current production ({current_r2:.3f})"
                    )

        is_ready = len(validation_issues) == 0
        logger.info(f"Deployment readiness: {is_ready}, issues: {len(validation_issues)}")

        return is_ready, validation_issues

    def deploy_to_staging(
        self,
        deployment_plan: DeploymentPlan
    ) -> Dict[str, Any]:
        """스테이징 환경에 배포."""

        logger.info(f"Deploying {deployment_plan.source_model.model_name} to staging")

        try:
            # 검증 수행
            if deployment_plan.validation_required:
                is_ready, issues = self.validate_deployment_readiness(deployment_plan)
                if not is_ready:
                    return {
                        "status": "failed",
                        "reason": "validation_failed",
                        "issues": issues
                    }

            # 모델을 스테이징에 등록
            model = deployment_plan.source_model
            model.stage = DeploymentStage.STAGING
            model.deployment_timestamp = datetime.now(ZoneInfo("Asia/Seoul"))

            # MLflow에서 스테이징으로 이동
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model.model_name,
                version=model.version,
                stage="Staging"
            )

            # S3에 스테이징 모델 저장
            s3_key = self._store_staged_model(model)

            # 배포된 모델 목록에 추가
            self.deployed_models[DeploymentStage.STAGING].append(model)

            # 오래된 스테이징 모델 정리
            self._cleanup_old_staging_models()

            # WandB 로깅
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    "deployment/staging_success": 1.0,
                    "deployment/staging_count": len(self.deployed_models[DeploymentStage.STAGING]),
                    "deployment/model_performance_r2": model.performance_metrics.get("r2", 0.0)
                })

            logger.info(f"Successfully deployed {model.model_name} v{model.version} to staging")

            return {
                "status": "success",
                "stage": "staging",
                "model_version": model,
                "s3_key": s3_key,
                "deployment_timestamp": model.deployment_timestamp
            }

        except Exception as e:
            logger.error(f"Failed to deploy to staging: {e}")

            if self.wandb_manager:
                self.wandb_manager.log_metrics({"deployment/staging_failure": 1.0})

            return {
                "status": "failed",
                "reason": "deployment_error",
                "error": str(e)
            }

    def deploy_canary(
        self,
        deployment_plan: DeploymentPlan
    ) -> Dict[str, Any]:
        """카나리 배포 실행."""

        logger.info(f"Starting canary deployment for {deployment_plan.source_model.model_name}")

        try:
            model = deployment_plan.source_model
            model.stage = DeploymentStage.CANARY
            model.deployment_timestamp = datetime.now(ZoneInfo("Asia/Seoul"))

            # 카나리 트래픽 단계별 배포
            traffic_steps = deployment_plan.canary_traffic_steps or [10, 25, 50, 100]
            duration_per_step = deployment_plan.canary_duration_minutes // len(traffic_steps)

            canary_results = []

            for step_idx, traffic_percentage in enumerate(traffic_steps):
                logger.info(f"Canary step {step_idx + 1}/{len(traffic_steps)}: {traffic_percentage}% traffic")

                # 트래픽 비율 설정
                model.traffic_percentage = traffic_percentage

                # 트래픽 라우팅 설정 (실제 환경에서는 로드밸런서 설정)
                routing_config = self._configure_traffic_routing(model, traffic_percentage)

                # 성능 모니터링 기간
                monitoring_start = datetime.now(ZoneInfo("Asia/Seoul"))

                # 시뮬레이션된 모니터링 (실제로는 duration_per_step만큼 대기)
                step_results = self._monitor_canary_performance(
                    model,
                    monitoring_duration_minutes=max(1, duration_per_step)
                )

                canary_results.append({
                    "step": step_idx + 1,
                    "traffic_percentage": traffic_percentage,
                    "performance": step_results,
                    "timestamp": monitoring_start
                })

                # 성능 문제 감지시 롤백
                if step_results.get("performance_degradation", 0.0) > deployment_plan.rollback_threshold:
                    logger.warning(f"Performance degradation detected in canary step {step_idx + 1}")

                    rollback_result = self.rollback_deployment(model)

                    return {
                        "status": "rolled_back",
                        "reason": "performance_degradation",
                        "canary_results": canary_results,
                        "rollback_result": rollback_result
                    }

            # 카나리 배포 성공
            self.deployed_models[DeploymentStage.CANARY].append(model)

            # WandB 로깅
            if self.wandb_manager:
                avg_performance = sum(r["performance"].get("r2", 0) for r in canary_results) / len(canary_results)
                self.wandb_manager.log_metrics({
                    "deployment/canary_success": 1.0,
                    "deployment/canary_steps_completed": len(traffic_steps),
                    "deployment/canary_avg_performance": avg_performance
                })

            logger.info(f"Canary deployment completed successfully")

            return {
                "status": "success",
                "stage": "canary",
                "model_version": model,
                "canary_results": canary_results,
                "ready_for_production": True
            }

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")

            if self.wandb_manager:
                self.wandb_manager.log_metrics({"deployment/canary_failure": 1.0})

            return {
                "status": "failed",
                "reason": "canary_error",
                "error": str(e)
            }

    def promote_to_production(
        self,
        model_version: ModelVersion,
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    ) -> Dict[str, Any]:
        """프로덕션으로 승격."""

        logger.info(f"Promoting {model_version.model_name} v{model_version.version} to production")

        try:
            # 현재 프로덕션 모델 백업
            current_production = self._get_current_production_model()
            if current_production:
                current_production.stage = DeploymentStage.ROLLBACK
                self.deployed_models[DeploymentStage.ROLLBACK].append(current_production)

            # 새 모델을 프로덕션으로 설정
            model_version.stage = DeploymentStage.PRODUCTION
            model_version.traffic_percentage = 100.0
            model_version.deployment_timestamp = datetime.now(ZoneInfo("Asia/Seoul"))

            # MLflow에서 프로덕션으로 이동
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_version.model_name,
                version=model_version.version,
                stage="Production"
            )

            # S3에 프로덕션 모델 저장
            s3_key = self._store_production_model(model_version)

            # 배포된 모델 목록 업데이트
            self.deployed_models[DeploymentStage.PRODUCTION] = [model_version]

            # 카나리에서 제거
            self.deployed_models[DeploymentStage.CANARY] = [
                m for m in self.deployed_models[DeploymentStage.CANARY]
                if m.mlflow_run_id != model_version.mlflow_run_id
            ]

            # WandB 로깅
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    "deployment/production_promotion": 1.0,
                    "deployment/production_model_performance": model_version.performance_metrics.get("r2", 0.0)
                })

            logger.info(f"Successfully promoted to production: {model_version.model_name} v{model_version.version}")

            return {
                "status": "success",
                "stage": "production",
                "model_version": model_version,
                "s3_key": s3_key,
                "previous_production": current_production,
                "deployment_timestamp": model_version.deployment_timestamp
            }

        except Exception as e:
            logger.error(f"Production promotion failed: {e}")

            if self.wandb_manager:
                self.wandb_manager.log_metrics({"deployment/production_failure": 1.0})

            return {
                "status": "failed",
                "reason": "promotion_error",
                "error": str(e)
            }

    def rollback_deployment(
        self,
        failed_model: ModelVersion
    ) -> Dict[str, Any]:
        """배포 롤백 실행."""

        logger.warning(f"Rolling back deployment for {failed_model.model_name} v{failed_model.version}")

        try:
            # 롤백 대상 모델 찾기
            rollback_candidates = self.deployed_models[DeploymentStage.ROLLBACK]

            if not rollback_candidates:
                # 가장 최근 안정된 프로덕션 모델 찾기
                rollback_model = self._find_stable_fallback_model()
            else:
                rollback_model = rollback_candidates[-1]  # 가장 최근 백업

            if not rollback_model:
                raise ValueError("No stable model available for rollback")

            # 롤백 실행
            rollback_model.stage = DeploymentStage.PRODUCTION
            rollback_model.traffic_percentage = 100.0
            rollback_model.deployment_timestamp = datetime.now(ZoneInfo("Asia/Seoul"))

            # MLflow 업데이트
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=rollback_model.model_name,
                version=rollback_model.version,
                stage="Production"
            )

            # 배포 상태 업데이트
            self.deployed_models[DeploymentStage.PRODUCTION] = [rollback_model]

            # 실패한 모델 제거
            for stage in [DeploymentStage.CANARY, DeploymentStage.STAGING]:
                self.deployed_models[stage] = [
                    m for m in self.deployed_models[stage]
                    if m.mlflow_run_id != failed_model.mlflow_run_id
                ]

            # WandB 알림
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    "deployment/rollback_executed": 1.0,
                    "deployment/rollback_model_performance": rollback_model.performance_metrics.get("r2", 0.0)
                })

                self.wandb_manager.create_alert(
                    title="Model Deployment Rollback",
                    text=f"Rolled back to {rollback_model.model_name} v{rollback_model.version} "
                         f"due to issues with {failed_model.model_name} v{failed_model.version}",
                    level="WARN"
                )

            logger.info(f"Successfully rolled back to {rollback_model.model_name} v{rollback_model.version}")

            return {
                "status": "success",
                "rollback_model": rollback_model,
                "failed_model": failed_model,
                "rollback_timestamp": rollback_model.deployment_timestamp
            }

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

            if self.wandb_manager:
                self.wandb_manager.create_alert(
                    title="CRITICAL: Rollback Failed",
                    text=f"Failed to rollback deployment: {str(e)}",
                    level="ERROR"
                )

            return {
                "status": "failed",
                "reason": "rollback_error",
                "error": str(e)
            }

    def health_check_deployed_models(self) -> Dict[str, Any]:
        """배포된 모델들의 건강 상태 점검."""

        logger.info("Performing health check on deployed models")

        health_results = {}

        for stage, models in self.deployed_models.items():
            stage_results = []

            for model in models:
                try:
                    # 모델 상태 점검
                    health_status = self._check_model_health(model)
                    model.health_status = health_status["status"]
                    model.last_health_check = datetime.now(ZoneInfo("Asia/Seoul"))

                    stage_results.append({
                        "model": f"{model.model_name}_v{model.version}",
                        "health_status": health_status,
                        "last_check": model.last_health_check
                    })

                    # WandB 로깅
                    if self.wandb_manager:
                        self.wandb_manager.log_metrics({
                            f"health/{stage.value}_{model.model_name}_status": 1.0 if health_status["status"] == "healthy" else 0.0,
                            f"health/{stage.value}_{model.model_name}_response_time": health_status.get("response_time_ms", 0)
                        })

                except Exception as e:
                    logger.error(f"Health check failed for {model.model_name}: {e}")
                    stage_results.append({
                        "model": f"{model.model_name}_v{model.version}",
                        "health_status": {"status": "error", "error": str(e)},
                        "last_check": datetime.now(ZoneInfo("Asia/Seoul"))
                    })

            health_results[stage.value] = stage_results

        return health_results

    def _get_current_production_model(self) -> Optional[ModelVersion]:
        """현재 프로덕션 모델 반환."""
        production_models = self.deployed_models[DeploymentStage.PRODUCTION]
        return production_models[0] if production_models else None

    def _store_staged_model(self, model: ModelVersion) -> str:
        """스테이징 모델을 S3에 저장."""
        # 모델 바이너리 다운로드 및 S3 업로드 로직
        model_data = self._load_model_binary(model.mlflow_run_id)

        return self.s3_manager.store_model_artifact(
            model_data=model_data,
            model_name=model.model_name,
            version=model.version,
            stage="staging",
            metadata=model.metadata
        )

    def _store_production_model(self, model: ModelVersion) -> str:
        """프로덕션 모델을 S3에 저장."""
        model_data = self._load_model_binary(model.mlflow_run_id)

        return self.s3_manager.store_model_artifact(
            model_data=model_data,
            model_name=model.model_name,
            version=model.version,
            stage="production",
            metadata=model.metadata
        )

    def _load_model_binary(self, mlflow_run_id: str) -> bytes:
        """MLflow에서 모델 바이너리 로드."""
        try:
            model = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/model")
            import pickle
            return pickle.dumps(model)
        except Exception as e:
            logger.error(f"Failed to load model binary: {e}")
            raise

    def _configure_traffic_routing(self, model: ModelVersion, traffic_percentage: float) -> Dict[str, Any]:
        """트래픽 라우팅 설정 (시뮬레이션)."""
        # 실제 환경에서는 로드밸런서나 API 게이트웨이 설정
        return {
            "model_id": model.mlflow_run_id,
            "traffic_percentage": traffic_percentage,
            "timestamp": datetime.now(ZoneInfo("Asia/Seoul"))
        }

    def _monitor_canary_performance(
        self,
        model: ModelVersion,
        monitoring_duration_minutes: int
    ) -> Dict[str, Any]:
        """카나리 성능 모니터링 (시뮬레이션)."""
        # 실제 환경에서는 실시간 메트릭 수집

        # 시뮬레이션된 성능 데이터
        base_performance = model.performance_metrics.get("r2", 0.8)

        # 약간의 변동성 추가
        import random
        current_performance = base_performance * (0.95 + random.random() * 0.1)

        return {
            "r2": current_performance,
            "mae": 0.3 + random.random() * 0.1,
            "response_time_ms": 50 + random.randint(0, 50),
            "error_rate": random.random() * 0.01,
            "performance_degradation": max(0, base_performance - current_performance) / base_performance
        }

    def _check_model_health(self, model: ModelVersion) -> Dict[str, Any]:
        """모델 건강 상태 점검."""
        try:
            # 실제 환경에서는 모델 엔드포인트 호출 및 응답 시간 측정

            # 시뮬레이션된 건강 점검
            import random
            response_time = 50 + random.randint(0, 100)

            if response_time > 200:
                status = "unhealthy"
            elif response_time > 100:
                status = "degraded"
            else:
                status = "healthy"

            return {
                "status": status,
                "response_time_ms": response_time,
                "cpu_usage": random.uniform(10, 80),
                "memory_usage": random.uniform(20, 70),
                "last_prediction_timestamp": datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(minutes=random.randint(1, 10))
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _find_stable_fallback_model(self) -> Optional[ModelVersion]:
        """안정된 폴백 모델 찾기."""
        # MLflow에서 가장 최근의 안정된 프로덕션 모델 찾기
        try:
            client = mlflow.tracking.MlflowClient()
            # 실제 구현에서는 모델 레지스트리에서 이전 프로덕션 모델들을 조회
            return None  # 시뮬레이션
        except Exception:
            return None

    def _cleanup_old_staging_models(self):
        """오래된 스테이징 모델 정리."""
        max_staging = self.deployment_config["max_staging_models"]

        if len(self.deployed_models[DeploymentStage.STAGING]) > max_staging:
            # 오래된 모델 제거 (deployment_timestamp 기준)
            sorted_models = sorted(
                self.deployed_models[DeploymentStage.STAGING],
                key=lambda m: m.deployment_timestamp,
                reverse=True
            )

            self.deployed_models[DeploymentStage.STAGING] = sorted_models[:max_staging]

            logger.info(f"Cleaned up old staging models, kept {max_staging} most recent")