"""모델 평가 및 검증 시스템."""

from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from zoneinfo import ZoneInfo

from ..storage import S3StorageManager
from ..tracking import WandBManager

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """모델 성능 지표."""
    timestamp: datetime
    model_name: str
    model_version: str

    # 회귀 메트릭
    mae: float          # Mean Absolute Error
    mse: float          # Mean Squared Error
    rmse: float         # Root Mean Squared Error
    r2: float           # R-squared
    mape: float         # Mean Absolute Percentage Error

    # 예측 품질 메트릭
    prediction_confidence: float
    data_quality_score: float
    feature_quality_score: float

    # 운영 메트릭
    prediction_latency_ms: float
    model_size_mb: float

    # 메타데이터
    sample_count: int
    evaluation_period_hours: int


@dataclass
class ValidationResult:
    """모델 검증 결과."""
    is_valid: bool
    confidence_score: float

    performance_metrics: ModelPerformanceMetrics
    validation_issues: List[str]

    # 임계값 비교
    meets_accuracy_threshold: bool
    meets_latency_threshold: bool
    meets_stability_threshold: bool

    # 권장 사항
    recommendations: List[str]
    retraining_required: bool


class ModelEvaluator:
    """모델 평가 및 검증 시스템."""

    def __init__(
        self,
        s3_manager: S3StorageManager,
        mlflow_tracking_uri: Optional[str] = None,
        wandb_manager: Optional[WandBManager] = None
    ):
        self.s3_manager = s3_manager
        self.wandb_manager = wandb_manager

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # 성능 임계값 설정
        self.performance_thresholds = {
            "min_r2": 0.7,              # 최소 R²
            "max_mae": 0.5,             # 최대 평균 절대 오차
            "max_mape": 15.0,           # 최대 평균 절대 백분율 오차
            "min_confidence": 0.8,      # 최소 예측 신뢰도
            "max_latency_ms": 1000,     # 최대 응답 시간
            "min_data_quality": 70.0,   # 최소 데이터 품질
        }

        # 안정성 임계값
        self.stability_thresholds = {
            "max_performance_degradation": 0.1,  # 10% 성능 저하
            "min_evaluation_samples": 24,        # 최소 24시간 데이터
            "performance_consistency_std": 0.05,  # 성능 일관성
        }

    def evaluate_model_performance(
        self,
        model_name: str,
        model_version: str,
        predictions: List[float],
        actual_values: List[float],
        prediction_metadata: List[Dict[str, Any]],
        evaluation_period_hours: int = 24
    ) -> ModelPerformanceMetrics:
        """모델 성능 평가."""

        if len(predictions) != len(actual_values):
            raise ValueError("Predictions and actual values must have same length")

        if len(predictions) == 0:
            raise ValueError("Cannot evaluate with empty predictions")

        predictions = np.array(predictions)
        actual_values = np.array(actual_values)

        # 기본 회귀 메트릭 계산
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predictions)

        # MAPE 계산 (0으로 나누기 방지)
        mask = actual_values != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(actual_values[mask], predictions[mask])
        else:
            mape = float('inf')

        # 예측 품질 메트릭 계산
        avg_confidence = np.mean([
            meta.get('confidence', 0.0) for meta in prediction_metadata
        ])

        avg_data_quality = np.mean([
            meta.get('data_quality', 0.0) for meta in prediction_metadata
        ])

        avg_feature_quality = np.mean([
            meta.get('feature_quality', 0.0) for meta in prediction_metadata
        ])

        # 지연 시간 계산
        avg_latency = np.mean([
            meta.get('prediction_latency_ms', 0.0) for meta in prediction_metadata
        ])

        # 모델 크기 (추정)
        model_size = prediction_metadata[0].get('model_size_mb', 0.0) if prediction_metadata else 0.0

        return ModelPerformanceMetrics(
            timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
            model_name=model_name,
            model_version=model_version,
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            mape=mape,
            prediction_confidence=avg_confidence,
            data_quality_score=avg_data_quality,
            feature_quality_score=avg_feature_quality,
            prediction_latency_ms=avg_latency,
            model_size_mb=model_size,
            sample_count=len(predictions),
            evaluation_period_hours=evaluation_period_hours
        )

    def validate_model(
        self,
        performance_metrics: ModelPerformanceMetrics,
        historical_metrics: List[ModelPerformanceMetrics]
    ) -> ValidationResult:
        """모델 검증 수행."""

        validation_issues = []
        recommendations = []

        # 기본 성능 검증
        meets_accuracy = self._validate_accuracy(performance_metrics, validation_issues)
        meets_latency = self._validate_latency(performance_metrics, validation_issues)
        meets_stability = self._validate_stability(
            performance_metrics, historical_metrics, validation_issues
        )

        # 전체 신뢰도 점수 계산
        confidence_score = self._calculate_confidence_score(performance_metrics)

        # 재훈련 필요성 판단
        retraining_required = self._assess_retraining_need(
            performance_metrics, historical_metrics, recommendations
        )

        # 추가 권장사항 생성
        self._generate_recommendations(performance_metrics, recommendations)

        is_valid = (
            meets_accuracy and
            meets_latency and
            meets_stability and
            confidence_score > 0.7
        )

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            performance_metrics=performance_metrics,
            validation_issues=validation_issues,
            meets_accuracy_threshold=meets_accuracy,
            meets_latency_threshold=meets_latency,
            meets_stability_threshold=meets_stability,
            recommendations=recommendations,
            retraining_required=retraining_required
        )

    def _validate_accuracy(
        self,
        metrics: ModelPerformanceMetrics,
        issues: List[str]
    ) -> bool:
        """정확도 검증."""

        meets_requirements = True

        if metrics.r2 < self.performance_thresholds["min_r2"]:
            issues.append(f"R² score ({metrics.r2:.3f}) below threshold ({self.performance_thresholds['min_r2']})")
            meets_requirements = False

        if metrics.mae > self.performance_thresholds["max_mae"]:
            issues.append(f"MAE ({metrics.mae:.3f}) above threshold ({self.performance_thresholds['max_mae']})")
            meets_requirements = False

        if metrics.mape > self.performance_thresholds["max_mape"]:
            issues.append(f"MAPE ({metrics.mape:.1f}%) above threshold ({self.performance_thresholds['max_mape']}%)")
            meets_requirements = False

        if metrics.prediction_confidence < self.performance_thresholds["min_confidence"]:
            issues.append(f"Prediction confidence ({metrics.prediction_confidence:.3f}) below threshold")
            meets_requirements = False

        return meets_requirements

    def _validate_latency(
        self,
        metrics: ModelPerformanceMetrics,
        issues: List[str]
    ) -> bool:
        """지연 시간 검증."""

        if metrics.prediction_latency_ms > self.performance_thresholds["max_latency_ms"]:
            issues.append(
                f"Prediction latency ({metrics.prediction_latency_ms:.0f}ms) "
                f"above threshold ({self.performance_thresholds['max_latency_ms']}ms)"
            )
            return False

        return True

    def _validate_stability(
        self,
        current_metrics: ModelPerformanceMetrics,
        historical_metrics: List[ModelPerformanceMetrics],
        issues: List[str]
    ) -> bool:
        """안정성 검증."""

        if len(historical_metrics) < self.stability_thresholds["min_evaluation_samples"]:
            issues.append(f"Insufficient historical data for stability assessment ({len(historical_metrics)} samples)")
            return False

        # 최근 성능 추세 분석
        recent_r2_scores = [m.r2 for m in historical_metrics[-7:]]  # 최근 7개 데이터

        if len(recent_r2_scores) > 2:
            r2_std = np.std(recent_r2_scores)
            if r2_std > self.stability_thresholds["performance_consistency_std"]:
                issues.append(f"High performance variance detected (R² std: {r2_std:.3f})")
                return False

        # 성능 저하 감지
        if len(historical_metrics) >= 7:
            baseline_r2 = np.mean([m.r2 for m in historical_metrics[-7:-3]])
            recent_r2 = np.mean([m.r2 for m in historical_metrics[-3:]])

            degradation = (baseline_r2 - recent_r2) / baseline_r2
            if degradation > self.stability_thresholds["max_performance_degradation"]:
                issues.append(f"Performance degradation detected ({degradation:.1%})")
                return False

        return True

    def _calculate_confidence_score(self, metrics: ModelPerformanceMetrics) -> float:
        """종합 신뢰도 점수 계산."""

        # 각 지표의 정규화된 점수 계산
        r2_score = min(metrics.r2 / self.performance_thresholds["min_r2"], 1.0)
        mae_score = max(0, 1 - metrics.mae / self.performance_thresholds["max_mae"])
        mape_score = max(0, 1 - metrics.mape / self.performance_thresholds["max_mape"])
        confidence_score = metrics.prediction_confidence / self.performance_thresholds["min_confidence"]
        data_quality_score = metrics.data_quality_score / 100.0

        # 가중 평균 계산
        weights = {
            'r2': 0.25,
            'mae': 0.20,
            'mape': 0.20,
            'confidence': 0.20,
            'data_quality': 0.15
        }

        total_score = (
            weights['r2'] * r2_score +
            weights['mae'] * mae_score +
            weights['mape'] * mape_score +
            weights['confidence'] * confidence_score +
            weights['data_quality'] * data_quality_score
        )

        return min(max(total_score, 0.0), 1.0)

    def _assess_retraining_need(
        self,
        current_metrics: ModelPerformanceMetrics,
        historical_metrics: List[ModelPerformanceMetrics],
        recommendations: List[str]
    ) -> bool:
        """재훈련 필요성 평가."""

        retraining_needed = False

        # 기본 성능 임계값 미달
        if (current_metrics.r2 < self.performance_thresholds["min_r2"] or
            current_metrics.mae > self.performance_thresholds["max_mae"]):
            recommendations.append("재훈련 권장: 기본 성능 임계값 미달")
            retraining_needed = True

        # 지속적 성능 저하
        if len(historical_metrics) >= 7:
            recent_trend = [m.r2 for m in historical_metrics[-7:]]
            if len(recent_trend) > 3:
                slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
                if slope < -0.01:  # 1% 이상 하락 추세
                    recommendations.append("재훈련 권장: 지속적 성능 저하 감지")
                    retraining_needed = True

        # 데이터 품질 문제
        if current_metrics.data_quality_score < self.performance_thresholds["min_data_quality"]:
            recommendations.append("데이터 수집 시스템 점검 필요")

        return retraining_needed

    def _generate_recommendations(
        self,
        metrics: ModelPerformanceMetrics,
        recommendations: List[str]
    ):
        """성능 개선 권장사항 생성."""

        if metrics.prediction_latency_ms > 500:
            recommendations.append("모델 최적화 권장: 예측 지연 시간 개선 필요")

        if metrics.feature_quality_score < 80:
            recommendations.append("피처 엔지니어링 개선 권장: 피처 품질 향상 필요")

        if metrics.prediction_confidence < 0.9:
            recommendations.append("모델 신뢰도 개선: 불확실성 추정 기법 적용 고려")

        if metrics.sample_count < 24:
            recommendations.append("더 많은 평가 데이터 수집 권장")

    def store_evaluation_results(
        self,
        validation_result: ValidationResult,
        experiment_name: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ) -> str:
        """평가 결과 저장."""

        metrics = validation_result.performance_metrics

        # MLflow에 메트릭 로깅
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"evaluation_{metrics.model_name}_{metrics.model_version}"):
            # 성능 메트릭
            mlflow.log_metric("mae", metrics.mae)
            mlflow.log_metric("mse", metrics.mse)
            mlflow.log_metric("rmse", metrics.rmse)
            mlflow.log_metric("r2", metrics.r2)
            mlflow.log_metric("mape", metrics.mape)

            # 품질 메트릭
            mlflow.log_metric("prediction_confidence", metrics.prediction_confidence)
            mlflow.log_metric("data_quality_score", metrics.data_quality_score)
            mlflow.log_metric("feature_quality_score", metrics.feature_quality_score)

            # 운영 메트릭
            mlflow.log_metric("prediction_latency_ms", metrics.prediction_latency_ms)
            mlflow.log_metric("model_size_mb", metrics.model_size_mb)

            # 검증 결과
            mlflow.log_metric("confidence_score", validation_result.confidence_score)
            mlflow.log_metric("is_valid", 1.0 if validation_result.is_valid else 0.0)
            mlflow.log_metric("retraining_required", 1.0 if validation_result.retraining_required else 0.0)

            # 메타데이터
            mlflow.log_param("model_name", metrics.model_name)
            mlflow.log_param("model_version", metrics.model_version)
            mlflow.log_param("evaluation_period_hours", metrics.evaluation_period_hours)
            mlflow.log_param("sample_count", metrics.sample_count)

            # 검증 이슈 및 권장사항
            if validation_result.validation_issues:
                mlflow.log_text("\n".join(validation_result.validation_issues), "validation_issues.txt")

            if validation_result.recommendations:
                mlflow.log_text("\n".join(validation_result.recommendations), "recommendations.txt")

        # WandB에 메트릭 로깅
        if self.wandb_manager:
            if not wandb_run_name:
                wandb_run_name = f"evaluation_{metrics.model_name}_{metrics.model_version}"

            # 평가 전용 실행 시작
            self.wandb_manager.start_run(
                run_name=wandb_run_name,
                job_type="evaluation",
                config={
                    "model_name": metrics.model_name,
                    "model_version": metrics.model_version,
                    "evaluation_period_hours": metrics.evaluation_period_hours,
                    "sample_count": metrics.sample_count,
                    "performance_thresholds": self.performance_thresholds
                },
                tags=["evaluation", "model-validation", metrics.model_name]
            )

            # 성능 메트릭 로깅
            performance_metrics = {
                "mae": metrics.mae,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "mape": metrics.mape,
                "prediction_confidence": metrics.prediction_confidence,
                "data_quality_score": metrics.data_quality_score,
                "feature_quality_score": metrics.feature_quality_score,
                "prediction_latency_ms": metrics.prediction_latency_ms,
                "model_size_mb": metrics.model_size_mb
            }

            self.wandb_manager.log_model_performance(
                performance_metrics,
                metrics.model_name,
                metrics.model_version
            )

            # 검증 결과 로깅
            validation_metrics = {
                "confidence_score": validation_result.confidence_score,
                "is_valid": validation_result.is_valid,
                "retraining_required": validation_result.retraining_required,
                "meets_accuracy_threshold": validation_result.meets_accuracy_threshold,
                "meets_latency_threshold": validation_result.meets_latency_threshold,
                "meets_stability_threshold": validation_result.meets_stability_threshold,
                "validation_issues_count": len(validation_result.validation_issues),
                "recommendations_count": len(validation_result.recommendations)
            }

            self.wandb_manager.log_metrics(validation_metrics)

            # 평가 보고서 생성 및 로깅
            evaluation_report = self.generate_performance_report(validation_result, [])
            self.wandb_manager.log_evaluation_report(
                evaluation_report,
                metrics.model_name
            )

            # 임계값 비교 시각화
            threshold_comparison = {
                "r2_vs_threshold": [metrics.r2, self.performance_thresholds["min_r2"]],
                "mae_vs_threshold": [metrics.mae, self.performance_thresholds["max_mae"]],
                "mape_vs_threshold": [metrics.mape, self.performance_thresholds["max_mape"]],
                "latency_vs_threshold": [metrics.prediction_latency_ms, self.performance_thresholds["max_latency_ms"]]
            }

            for metric_name, values in threshold_comparison.items():
                self.wandb_manager.log_metrics({
                    f"threshold_comparison/{metric_name}_current": values[0],
                    f"threshold_comparison/{metric_name}_threshold": values[1]
                })

            # 알림 생성 (필요시)
            if validation_result.retraining_required:
                self.wandb_manager.create_alert(
                    title="Model Retraining Required",
                    text=f"Model {metrics.model_name} v{metrics.model_version} requires retraining. "
                         f"Issues: {', '.join(validation_result.validation_issues[:3])}",
                    level="WARN"
                )

            if not validation_result.is_valid:
                self.wandb_manager.create_alert(
                    title="Model Validation Failed",
                    text=f"Model {metrics.model_name} v{metrics.model_version} failed validation. "
                         f"Confidence score: {validation_result.confidence_score:.3f}",
                    level="ERROR"
                )

            # 실행 완료
            self.wandb_manager.finish_run()

        # S3에 상세 결과 저장
        evaluation_data = {
            "validation_result": validation_result,
            "timestamp": metrics.timestamp.isoformat(),
            "model_info": {
                "name": metrics.model_name,
                "version": metrics.model_version
            }
        }

        s3_key = self.s3_manager.store_evaluation_results(
            evaluation_data,
            metrics.model_name,
            metrics.model_version,
            metrics.timestamp
        )

        logger.info(f"Evaluation results stored: MLflow + S3 ({s3_key})")
        return s3_key

    def get_historical_metrics(
        self,
        model_name: str,
        days_back: int = 7
    ) -> List[ModelPerformanceMetrics]:
        """과거 성능 메트릭 조회."""

        try:
            # MLflow에서 과거 실행 결과 조회
            experiment = mlflow.get_experiment_by_name("model_evaluation")
            if not experiment:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_name = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=days_back * 24  # 시간당 1개씩 최대
            )

            historical_metrics = []
            for _, run in runs.iterrows():
                try:
                    metrics = ModelPerformanceMetrics(
                        timestamp=pd.to_datetime(run['start_time']),
                        model_name=run['params.model_name'],
                        model_version=run['params.model_version'],
                        mae=run['metrics.mae'],
                        mse=run['metrics.mse'],
                        rmse=run['metrics.rmse'],
                        r2=run['metrics.r2'],
                        mape=run['metrics.mape'],
                        prediction_confidence=run['metrics.prediction_confidence'],
                        data_quality_score=run['metrics.data_quality_score'],
                        feature_quality_score=run['metrics.feature_quality_score'],
                        prediction_latency_ms=run['metrics.prediction_latency_ms'],
                        model_size_mb=run['metrics.model_size_mb'],
                        sample_count=int(run['params.sample_count']),
                        evaluation_period_hours=int(run['params.evaluation_period_hours'])
                    )
                    historical_metrics.append(metrics)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse historical metric: {e}")
                    continue

            return historical_metrics

        except Exception as e:
            logger.error(f"Failed to retrieve historical metrics: {e}")
            return []

    def generate_performance_report(
        self,
        validation_result: ValidationResult,
        historical_metrics: List[ModelPerformanceMetrics]
    ) -> Dict[str, Any]:
        """성능 보고서 생성."""

        metrics = validation_result.performance_metrics

        # 성능 추세 분석
        trend_analysis = self._analyze_performance_trend(historical_metrics)

        # 비교 분석
        comparison = self._compare_with_baseline(metrics, historical_metrics)

        report = {
            "evaluation_summary": {
                "model_name": metrics.model_name,
                "model_version": metrics.model_version,
                "timestamp": metrics.timestamp.isoformat(),
                "is_valid": validation_result.is_valid,
                "confidence_score": validation_result.confidence_score,
                "retraining_required": validation_result.retraining_required
            },
            "performance_metrics": {
                "accuracy": {
                    "r2": metrics.r2,
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                    "mape": metrics.mape
                },
                "quality": {
                    "prediction_confidence": metrics.prediction_confidence,
                    "data_quality_score": metrics.data_quality_score,
                    "feature_quality_score": metrics.feature_quality_score
                },
                "operational": {
                    "prediction_latency_ms": metrics.prediction_latency_ms,
                    "model_size_mb": metrics.model_size_mb,
                    "sample_count": metrics.sample_count
                }
            },
            "validation_results": {
                "meets_accuracy_threshold": validation_result.meets_accuracy_threshold,
                "meets_latency_threshold": validation_result.meets_latency_threshold,
                "meets_stability_threshold": validation_result.meets_stability_threshold,
                "validation_issues": validation_result.validation_issues,
                "recommendations": validation_result.recommendations
            },
            "trend_analysis": trend_analysis,
            "comparison_analysis": comparison,
            "thresholds": self.performance_thresholds
        }

        return report

    def _analyze_performance_trend(
        self,
        historical_metrics: List[ModelPerformanceMetrics]
    ) -> Dict[str, Any]:
        """성능 추세 분석."""

        if len(historical_metrics) < 3:
            return {"status": "insufficient_data", "message": "Not enough historical data for trend analysis"}

        # 최근 7일 R² 추세
        recent_r2 = [m.r2 for m in historical_metrics[-7:]]
        recent_mae = [m.mae for m in historical_metrics[-7:]]

        # 선형 추세 계산
        if len(recent_r2) > 2:
            r2_slope = np.polyfit(range(len(recent_r2)), recent_r2, 1)[0]
            mae_slope = np.polyfit(range(len(recent_mae)), recent_mae, 1)[0]

            trend_status = "stable"
            if r2_slope < -0.01:
                trend_status = "declining"
            elif r2_slope > 0.01:
                trend_status = "improving"

            return {
                "status": "analyzed",
                "r2_trend": {
                    "slope": r2_slope,
                    "status": trend_status,
                    "recent_values": recent_r2
                },
                "mae_trend": {
                    "slope": mae_slope,
                    "recent_values": recent_mae
                }
            }

        return {"status": "insufficient_data"}

    def _compare_with_baseline(
        self,
        current_metrics: ModelPerformanceMetrics,
        historical_metrics: List[ModelPerformanceMetrics]
    ) -> Dict[str, Any]:
        """기준선 대비 비교."""

        if len(historical_metrics) < 7:
            return {"status": "insufficient_baseline_data"}

        # 최근 7일 평균을 기준선으로 사용
        baseline_r2 = np.mean([m.r2 for m in historical_metrics[-7:]])
        baseline_mae = np.mean([m.mae for m in historical_metrics[-7:]])

        r2_change = ((current_metrics.r2 - baseline_r2) / baseline_r2) * 100
        mae_change = ((current_metrics.mae - baseline_mae) / baseline_mae) * 100

        return {
            "status": "compared",
            "baseline_period": "7_days",
            "r2_comparison": {
                "current": current_metrics.r2,
                "baseline": baseline_r2,
                "change_percent": r2_change
            },
            "mae_comparison": {
                "current": current_metrics.mae,
                "baseline": baseline_mae,
                "change_percent": mae_change
            }
        }