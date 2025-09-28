"""MLflow 통합 모듈."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient

from .models import BaseCommuteModel, TrainingResult, ModelFactory, compare_models
from ..storage.s3_manager import S3StorageManager

logger = logging.getLogger(__name__)


class MLflowManager:
    """MLflow 실험 추적 및 모델 관리"""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "commute-weather-prediction",
        s3_manager: Optional[S3StorageManager] = None
    ):
        """
        MLflow 매니저 초기화

        Args:
            tracking_uri: MLflow 추적 URI (None이면 로컬 파일 시스템)
            experiment_name: 실험 이름
            s3_manager: S3 스토리지 매니저 (모델 아티팩트 저장용)
        """
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.s3_manager = s3_manager

        # MLflow 설정
        mlflow.set_tracking_uri(self.tracking_uri)

        # 실험 설정
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)

            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set: {experiment_name}")

        except Exception as e:
            logger.error(f"Failed to set up MLflow experiment: {e}")
            raise

        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def log_training_run(
        self,
        model: BaseCommuteModel,
        training_result: TrainingResult,
        dataset_info: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        훈련 실행 로깅

        Args:
            model: 훈련된 모델
            training_result: 훈련 결과
            dataset_info: 데이터셋 정보
            run_name: 실행 이름
            tags: 추가 태그

        Returns:
            run_id: MLflow 실행 ID
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{training_result.model_type}_{timestamp}"

        with mlflow.start_run(run_name=run_name) as run:
            # 기본 정보 로깅
            mlflow.log_param("model_type", training_result.model_type)
            mlflow.log_param("training_data_size", training_result.training_data_size)
            mlflow.log_param("validation_data_size", training_result.validation_data_size)
            mlflow.log_param("feature_count", len(training_result.feature_names))

            # 하이퍼파라미터 로깅
            for param, value in training_result.hyperparameters.items():
                mlflow.log_param(f"hp_{param}", value)

            # 피처 정보 로깅
            mlflow.log_param("features", ",".join(training_result.feature_names))

            # 성능 메트릭 로깅
            metrics = training_result.metrics
            mlflow.log_metric("rmse", metrics.rmse)
            mlflow.log_metric("mae", metrics.mae)
            mlflow.log_metric("r2_score", metrics.r2)

            if metrics.cv_mean is not None:
                mlflow.log_metric("cv_rmse_mean", metrics.cv_mean)
                mlflow.log_metric("cv_rmse_std", metrics.cv_std or 0.0)

            # 데이터셋 정보 로깅
            if dataset_info:
                for key, value in dataset_info.items():
                    mlflow.log_param(f"data_{key}", value)

            # 스케일러 정보 로깅
            mlflow.log_param("uses_scaler", training_result.scaler is not None)

            # 모델 아티팩트 로깅
            try:
                mlflow.sklearn.log_model(
                    sk_model=training_result.model,
                    artifact_path="model",
                    registered_model_name=f"commute-weather-{training_result.model_type}"
                )

                # 스케일러도 별도로 저장
                if training_result.scaler is not None:
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.joblib') as f:
                        import joblib
                        joblib.dump(training_result.scaler, f.name)
                        mlflow.log_artifact(f.name, "scaler")
                        os.unlink(f.name)

                logger.info(f"Model artifacts logged successfully")

            except Exception as e:
                logger.warning(f"Failed to log model artifacts: {e}")

            # S3에 모델 저장
            if self.s3_manager:
                try:
                    self._save_model_to_s3(model, training_result, run.info.run_id)
                except Exception as e:
                    logger.warning(f"Failed to save model to S3: {e}")

            # 태그 설정
            if tags:
                mlflow.set_tags(tags)

            # 기본 태그 설정
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("domain", "weather-prediction")
            mlflow.set_tag("model_family", "regression")

            run_id = run.info.run_id
            logger.info(f"Training run logged: {run_id}")

            return run_id

    def _save_model_to_s3(
        self,
        model: BaseCommuteModel,
        training_result: TrainingResult,
        run_id: str
    ) -> str:
        """S3에 모델 저장"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            model.save_model(f.name)

            with open(f.name, 'rb') as model_file:
                model_data = model_file.read()

            metadata = {
                "model_type": training_result.model_type,
                "run_id": run_id,
                "rmse": str(training_result.metrics.rmse),
                "mae": str(training_result.metrics.mae),
                "r2_score": str(training_result.metrics.r2),
                "features": ",".join(training_result.feature_names),
                "training_size": str(training_result.training_data_size),
            }

            s3_key = self.s3_manager.store_model_artifact(
                model_data=model_data,
                model_name="commute-weather",
                version=f"run_{run_id}",
                stage="experiments",
                metadata=metadata
            )

            os.unlink(f.name)
            logger.info(f"Model saved to S3: {s3_key}")
            return s3_key

    def log_model_comparison(
        self,
        results: Dict[str, TrainingResult],
        dataset_info: Optional[Dict[str, Any]] = None,
        comparison_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        모델 비교 결과 로깅

        Args:
            results: 모델별 훈련 결과
            dataset_info: 데이터셋 정보
            comparison_name: 비교 실험 이름

        Returns:
            모델별 run_id 딕셔너리
        """
        if comparison_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_name = f"model_comparison_{timestamp}"

        run_ids = {}

        # 각 모델별로 개별 실행 로깅
        for model_type, result in results.items():
            model = ModelFactory.create_model(model_type)
            model.model = result.model
            model.scaler = result.scaler
            model.feature_names = result.feature_names
            model.is_fitted = True

            run_name = f"{comparison_name}_{model_type}"
            tags = {
                "comparison_experiment": comparison_name,
                "is_comparison": "true"
            }

            run_id = self.log_training_run(
                model=model,
                training_result=result,
                dataset_info=dataset_info,
                run_name=run_name,
                tags=tags
            )

            run_ids[model_type] = run_id

        logger.info(f"Model comparison logged: {comparison_name}")
        return run_ids

    def get_best_model_from_experiment(
        self,
        metric_name: str = "rmse",
        ascending: bool = True,
        max_results: int = 10
    ) -> Optional[Tuple[Run, str]]:
        """
        실험에서 최고 성능 모델 조회

        Args:
            metric_name: 정렬 기준 메트릭
            ascending: 오름차순 정렬 여부
            max_results: 최대 결과 수

        Returns:
            (최고 성능 실행, 모델 URI) 튜플
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=max_results
            )

            if runs.empty:
                logger.warning("No runs found in experiment")
                return None

            best_run_id = runs.iloc[0]['run_id']
            best_run = self.client.get_run(best_run_id)

            # 모델 URI 생성
            model_uri = f"runs:/{best_run_id}/model"

            logger.info(
                f"Best model found: {best_run_id} "
                f"({metric_name}: {runs.iloc[0][f'metrics.{metric_name}']})"
            )

            return best_run, model_uri

        except Exception as e:
            logger.error(f"Failed to retrieve best model: {e}")
            return None

    def load_model_from_run(self, run_id: str) -> BaseCommuteModel:
        """실행에서 모델 로드"""
        try:
            # MLflow에서 모델 로드
            model_uri = f"runs:/{run_id}/model"
            sklearn_model = mlflow.sklearn.load_model(model_uri)

            # 실행 정보 가져오기
            run = self.client.get_run(run_id)
            model_type = run.data.params.get("model_type", "linear")

            # 모델 인스턴스 생성
            model = ModelFactory.create_model(model_type)
            model.model = sklearn_model
            model.is_fitted = True

            # 피처 이름 복원
            features_param = run.data.params.get("features", "")
            if features_param:
                model.feature_names = features_param.split(",")

            # 스케일러 로드 (있는 경우)
            try:
                artifacts = self.client.list_artifacts(run_id, "scaler")
                if artifacts:
                    scaler_path = self.client.download_artifacts(run_id, "scaler")
                    import joblib
                    scaler_file = Path(scaler_path) / artifacts[0].path
                    model.scaler = joblib.load(scaler_file)
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
                model.scaler = None

            logger.info(f"Model loaded from run: {run_id}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from run {run_id}: {e}")
            raise

    def promote_model_to_stage(
        self,
        model_name: str,
        version: Union[int, str],
        stage: str
    ) -> None:
        """모델을 특정 스테이지로 승격"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} promoted to {stage}")

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise

    def get_production_model(self, model_name: str) -> Optional[BaseCommuteModel]:
        """프로덕션 스테이지의 모델 가져오기"""
        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )

            if not versions:
                logger.warning(f"No production version found for {model_name}")
                return None

            latest_version = versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"

            sklearn_model = mlflow.sklearn.load_model(model_uri)

            # 모델 메타데이터에서 타입 정보 가져오기
            model_type = latest_version.tags.get("model_type", "linear")

            model = ModelFactory.create_model(model_type)
            model.model = sklearn_model
            model.is_fitted = True

            logger.info(f"Production model loaded: {model_name} v{latest_version.version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

    def log_inference_metrics(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        model_version: Optional[str] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """추론 메트릭 로깅"""
        metrics = {
            "prediction_count": len(predictions),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
        }

        # 실제값이 있으면 성능 메트릭 계산
        if actuals is not None and len(actuals) == len(predictions):
            import numpy as np
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

            metrics.update({
                "inference_rmse": rmse,
                "inference_mae": mae,
                "inference_r2": r2,
            })

        # 추가 메트릭
        if additional_metrics:
            metrics.update(additional_metrics)

        # S3에 메트릭 저장
        if self.s3_manager:
            try:
                self.s3_manager.store_inference_metrics(
                    metrics=metrics,
                    model_name="commute-weather",
                    timestamp=datetime.now()
                )
            except Exception as e:
                logger.warning(f"Failed to store inference metrics to S3: {e}")

        logger.info(f"Inference metrics logged: {len(predictions)} predictions")

    def cleanup_old_experiments(self, keep_days: int = 30) -> None:
        """오래된 실험 정리"""
        try:
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=f"attribute.start_time < {cutoff_time * 1000}",  # MLflow는 밀리초 단위
                output_format="list"
            )

            for run in runs:
                # 프로덕션이나 스테이징 모델은 보존
                tags = run.data.tags
                if tags.get("stage") in ["Production", "Staging"]:
                    continue

                self.client.delete_run(run.info.run_id)
                logger.info(f"Deleted old run: {run.info.run_id}")

            logger.info(f"Cleanup completed: removed runs older than {keep_days} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old experiments: {e}")


def setup_mlflow_tracking(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "commute-weather-prediction"
) -> MLflowManager:
    """MLflow 추적 설정"""
    if tracking_uri is None:
        # 환경변수에서 확인
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    manager = MLflowManager(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    return manager