"""향상된 모델 훈련 시스템 - KMA 데이터 기반 피처 활용."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .models import TrainingResult, ModelMetrics, BaseCommuteModel
from .mlflow_integration import MLflowManager
from ..features.feature_engineering import WeatherFeatureEngineer
from ..data_sources.weather_api import WeatherObservation
from ..storage.s3_manager import S3StorageManager

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingConfig:
    """향상된 훈련 설정"""

    # 데이터 설정
    min_training_samples: int = 100
    test_size: float = 0.2
    validation_split: float = 0.2

    # 시계열 설정
    use_time_series_split: bool = True
    n_splits: int = 5

    # 피처 설정
    feature_selection_threshold: float = 0.01  # 피처 중요도 임계값
    max_features: int = 50  # 최대 피처 수

    # 모델 설정
    random_state: int = 42
    cross_validation: bool = True

    # 고급 설정
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = True
    early_stopping: bool = True


class EnhancedCommutePredictor:
    """KMA 데이터 기반 향상된 출퇴근 예측 모델"""

    def __init__(
        self,
        config: Optional[EnhancedTrainingConfig] = None,
        mlflow_manager: Optional[MLflowManager] = None,
        s3_manager: Optional[S3StorageManager] = None
    ):
        self.config = config or EnhancedTrainingConfig()
        self.mlflow_manager = mlflow_manager
        self.s3_manager = s3_manager

        self.feature_engineer = WeatherFeatureEngineer()
        self.scaler = StandardScaler()

        # 훈련된 모델들
        self.models: Dict[str, BaseCommuteModel] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []

        logger.info("Enhanced commute predictor initialized")

    def prepare_training_data(
        self,
        observations_list: List[List[WeatherObservation]],
        comfort_scores: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """KMA 관측 데이터를 훈련용 피처로 변환"""

        logger.info(f"Preparing training data from {len(observations_list)} observation sets")

        if len(observations_list) != len(comfort_scores):
            raise ValueError("Observations and comfort scores must have same length")

        feature_rows = []
        valid_scores = []

        for i, (observations, score) in enumerate(zip(observations_list, comfort_scores)):
            try:
                # 피처 엔지니어링 수행
                target_time = timestamps[i] if timestamps else observations[-1].timestamp
                features = self.feature_engineer.engineer_features(observations, target_time)

                # 피처 품질 확인
                quality_score = features.get('feature_quality_score', 0.0)
                if quality_score < 0.5:  # 낮은 품질 데이터 제외
                    logger.warning(f"Low quality features at index {i}: {quality_score:.3f}")
                    continue

                # 타겟값 추가
                features['comfort_score'] = score
                features['data_index'] = i

                feature_rows.append(features)
                valid_scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to engineer features for observation {i}: {e}")
                continue

        if not feature_rows:
            raise ValueError("No valid feature rows generated")

        # DataFrame 생성
        df = pd.DataFrame(feature_rows)

        # 무한값, NaN 처리
        df = df.replace([np.inf, -np.inf], np.nan)

        # 수치형 피처만 선택 (타겟 제외)
        numeric_features = []
        for col in df.columns:
            if col in ['comfort_score', 'data_index']:
                continue
            if df[col].dtype in ['int64', 'float64', 'bool']:
                numeric_features.append(col)

        self.feature_names = numeric_features
        logger.info(f"Generated {len(numeric_features)} numeric features")

        # 피처 데이터 준비
        X = df[numeric_features].fillna(0)  # NaN을 0으로 채움
        y = np.array(valid_scores)

        logger.info(f"Training data prepared: {len(X)} samples, {len(numeric_features)} features")
        return X, y

    def select_features(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """피처 선택 수행"""

        if not self.config.enable_feature_selection:
            return self.feature_names

        logger.info("Performing feature selection")

        # 기본 통계 기반 필터링
        feature_importance = {}

        for feature in X.columns:
            try:
                # 상관관계 기반 중요도
                correlation = abs(np.corrcoef(X[feature], y)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0

                # 분산 기반 중요도
                variance = X[feature].var()
                if np.isnan(variance):
                    variance = 0.0

                # 결합 중요도 점수
                importance = correlation * 0.7 + (variance / (variance + 1)) * 0.3
                feature_importance[feature] = importance

            except Exception as e:
                logger.warning(f"Failed to calculate importance for {feature}: {e}")
                feature_importance[feature] = 0.0

        # 임계값 기준 피처 선택
        selected_features = [
            feature for feature, importance in feature_importance.items()
            if importance >= self.config.feature_selection_threshold
        ]

        # 중요도 기준 상위 피처 선택
        if len(selected_features) > self.config.max_features:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_features = [f[0] for f in sorted_features[:self.config.max_features]]

        logger.info(f"Selected {len(selected_features)} features out of {len(self.feature_names)}")

        # 피처 중요도 로그
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 features:")
        for feature, importance in top_features:
            logger.info(f"  {feature}: {importance:.4f}")

        self.selected_features = selected_features
        return selected_features

    def train_enhanced_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, TrainingResult]:
        """향상된 모델들 훈련"""

        logger.info("Training enhanced models")

        # 피처 선택
        selected_features = self.select_features(X, y)
        X_selected = X[selected_features]

        # 데이터 분할
        if self.config.use_time_series_split:
            # 시계열 분할 (시간 순서 보존)
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            splits = list(tscv.split(X_selected))
            train_idx, val_idx = splits[-1]  # 마지막 분할 사용
        else:
            # 랜덤 분할
            train_idx, val_idx = train_test_split(
                range(len(X_selected)),
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 모델 설정
        models_config = {
            'ridge_enhanced': {
                'model': Ridge(
                    alpha=1.0,
                    random_state=self.config.random_state
                ),
                'hyperparameters': {'alpha': 1.0}
            },
            'random_forest_enhanced': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ),
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            },
            'gradient_boosting_enhanced': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=self.config.random_state
                ),
                'hyperparameters': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'subsample': 0.8
                }
            }
        }

        results = {}

        # 각 모델 훈련
        for model_name, config in models_config.items():
            try:
                logger.info(f"Training {model_name}")

                model = config['model']

                # 훈련
                if model_name.startswith('ridge'):
                    # Ridge는 스케일링된 데이터 사용
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                else:
                    # Tree 기반 모델은 원본 데이터 사용
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                # 성능 평가
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)

                # 교차 검증
                cv_scores = None
                cv_mean = None
                cv_std = None

                if self.config.cross_validation:
                    cv_scores = self._cross_validate_model(
                        model, X_selected, y, selected_features
                    )
                    if cv_scores is not None:
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()

                # 결과 생성
                metrics = ModelMetrics(
                    rmse=rmse,
                    mae=mae,
                    r2=r2,
                    cv_scores=cv_scores.tolist() if cv_scores is not None else None,
                    cv_mean=cv_mean,
                    cv_std=cv_std
                )

                result = TrainingResult(
                    model=model,
                    scaler=self.scaler if model_name.startswith('ridge') else None,
                    metrics=metrics,
                    feature_names=selected_features,
                    model_type=model_name,
                    hyperparameters=config['hyperparameters'],
                    training_data_size=len(X_train),
                    validation_data_size=len(X_val)
                )

                results[model_name] = result

                logger.info(
                    f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
                )

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # 최고 성능 모델 선택
        if results:
            best_model_name = min(results.keys(), key=lambda k: results[k].metrics.rmse)
            self.best_model_name = best_model_name
            logger.info(f"Best model: {best_model_name} (RMSE: {results[best_model_name].metrics.rmse:.4f})")

        return results

    def _cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Optional[np.ndarray]:
        """교차 검증 수행"""
        try:
            from sklearn.model_selection import cross_val_score

            X_features = X[feature_names]

            if hasattr(model, 'fit') and 'Ridge' in str(type(model)):
                # Ridge 모델은 스케일링 필요
                X_scaled = StandardScaler().fit_transform(X_features)
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=self.config.n_splits,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
            else:
                # Tree 기반 모델
                scores = cross_val_score(
                    model, X_features, y,
                    cv=self.config.n_splits,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )

            return -scores  # 양수로 변환

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return None

    def train_and_track(
        self,
        observations_list: List[List[WeatherObservation]],
        comfort_scores: List[float],
        timestamps: Optional[List[datetime]] = None,
        experiment_name: str = "enhanced-commute-prediction"
    ) -> Dict[str, str]:
        """훈련 및 MLflow 추적"""

        logger.info("Starting enhanced training with MLflow tracking")

        # 데이터 준비
        X, y = self.prepare_training_data(observations_list, comfort_scores, timestamps)

        # 모델 훈련
        results = self.train_enhanced_models(X, y)

        if not results:
            raise ValueError("No models were successfully trained")

        # MLflow 추적
        run_ids = {}
        if self.mlflow_manager:
            try:
                # 데이터셋 정보
                dataset_info = {
                    "total_samples": len(X),
                    "feature_count": len(self.selected_features),
                    "selected_features": ",".join(self.selected_features),
                    "feature_engineering_version": "kma_enhanced_v1",
                    "comfort_score_range": f"{y.min():.3f}-{y.max():.3f}",
                    "training_config": str(self.config.__dict__)
                }

                # 실험 설정
                self.mlflow_manager.experiment_name = experiment_name

                # 모델 비교 로깅
                comparison_run_ids = self.mlflow_manager.log_model_comparison(
                    results=results,
                    dataset_info=dataset_info,
                    comparison_name=f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

                run_ids.update(comparison_run_ids)

                logger.info(f"MLflow tracking completed: {len(run_ids)} runs logged")

            except Exception as e:
                logger.error(f"MLflow tracking failed: {e}")

        # S3 저장
        if self.s3_manager:
            try:
                self._save_training_artifacts_to_s3(results)
            except Exception as e:
                logger.warning(f"Failed to save to S3: {e}")

        return run_ids

    def _save_training_artifacts_to_s3(self, results: Dict[str, TrainingResult]) -> None:
        """훈련 아티팩트를 S3에 저장"""

        for model_name, result in results.items():
            try:
                # 모델 메타데이터
                metadata = {
                    "model_name": model_name,
                    "rmse": str(result.metrics.rmse),
                    "mae": str(result.metrics.mae),
                    "r2_score": str(result.metrics.r2),
                    "feature_count": str(len(result.feature_names)),
                    "training_size": str(result.training_data_size),
                    "feature_engineering": "kma_enhanced_v1",
                    "trained_at": datetime.now().isoformat()
                }

                # 모델 직렬화
                import joblib
                import io
                buffer = io.BytesIO()

                model_data = {
                    "model": result.model,
                    "scaler": result.scaler,
                    "feature_names": result.feature_names,
                    "config": self.config.__dict__
                }

                joblib.dump(model_data, buffer)
                model_bytes = buffer.getvalue()

                # S3 저장
                s3_key = self.s3_manager.store_model_artifact(
                    model_data=model_bytes,
                    model_name="enhanced-commute-weather",
                    version=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    stage="experiments",
                    metadata=metadata
                )

                logger.info(f"Model {model_name} saved to S3: {s3_key}")

            except Exception as e:
                logger.error(f"Failed to save {model_name} to S3: {e}")

    def predict_enhanced(
        self,
        observations: List[WeatherObservation],
        model_name: Optional[str] = None,
        target_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """향상된 예측 수행"""

        if not self.models and not model_name:
            raise ValueError("No trained models available")

        # 사용할 모델 결정
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        try:
            # 피처 엔지니어링
            features = self.feature_engineer.engineer_features(observations, target_time)

            # 선택된 피처만 사용
            X = pd.DataFrame([features])[self.selected_features].fillna(0)

            # 예측
            model = self.models[model_name]
            if hasattr(model, 'scaler') and model.scaler is not None:
                X_scaled = model.scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
            else:
                prediction = model.predict(X)[0]

            # 신뢰도 계산 (단순화된 버전)
            confidence = min(1.0, features.get('feature_quality_score', 0.8))

            result = {
                "prediction": float(prediction),
                "confidence": float(confidence),
                "model_used": model_name,
                "feature_quality": features.get('feature_quality_score', 0.0),
                "prediction_time": datetime.now().isoformat(),
                "features_count": len(self.selected_features)
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


def create_enhanced_trainer(
    mlflow_tracking_uri: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    config: Optional[EnhancedTrainingConfig] = None
) -> EnhancedCommutePredictor:
    """향상된 훈련기 생성"""

    # MLflow 매니저 설정
    mlflow_manager = None
    if mlflow_tracking_uri:
        mlflow_manager = MLflowManager(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="enhanced-commute-prediction"
        )

    # S3 매니저 설정
    s3_manager = None
    if s3_bucket:
        from ..config import ProjectPaths
        from pathlib import Path

        paths = ProjectPaths.from_root(Path.cwd())
        s3_manager = S3StorageManager(
            paths=paths,
            bucket=s3_bucket
        )

    return EnhancedCommutePredictor(
        config=config,
        mlflow_manager=mlflow_manager,
        s3_manager=s3_manager
    )