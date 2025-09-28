"""ML 모델 정의 및 훈련 로직."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """모델 성능 메트릭"""
    rmse: float
    mae: float
    r2: float
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


@dataclass
class TrainingResult:
    """모델 훈련 결과"""
    model: BaseEstimator
    scaler: Optional[StandardScaler]
    metrics: ModelMetrics
    feature_names: List[str]
    model_type: str
    hyperparameters: Dict[str, Any]
    training_data_size: int
    validation_data_size: int


class BaseCommuteModel(ABC):
    """출퇴근 예측 모델 베이스 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[BaseEstimator] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    @abstractmethod
    def create_model(self, **kwargs) -> BaseEstimator:
        """모델 인스턴스 생성"""
        pass

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """피처 준비 및 전처리"""
        feature_columns = [
            "temp_median_3h",
            "temp_range_3h",
            "wind_peak_3h",
            "precip_total_3h",
            "humidity_avg_3h",
        ]

        # 사용 가능한 피처만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        if not available_features:
            raise ValueError(f"No required features found in DataFrame. Available: {list(df.columns)}")

        self.feature_names = available_features
        X = df[available_features].copy()

        # 누락값 처리
        X = X.fillna(X.median())

        # 타겟 준비
        if "baseline_comfort_score" not in df.columns:
            raise ValueError("Target column 'baseline_comfort_score' not found")

        y = df["baseline_comfort_score"].values

        # 누락된 타겟 제거
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]

        return X.values, y

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        use_scaler: bool = True,
        cross_validation: bool = True,
        cv_folds: int = 5,
        **model_kwargs
    ) -> TrainingResult:
        """모델 훈련"""
        logger.info(f"Training {self.model_name} model")

        # 데이터 준비
        X, y = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid data available for training")

        logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")

        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 스케일링
        if use_scaler:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            self.scaler = None

        # 모델 생성 및 훈련
        self.model = self.create_model(**model_kwargs)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # 예측 및 평가
        y_pred = self.model.predict(X_val_scaled)

        # 기본 메트릭 계산
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 교차 검증
        cv_scores = None
        cv_mean = None
        cv_std = None

        if cross_validation and len(X) > cv_folds:
            try:
                # 전체 데이터에 대해 교차 검증
                if use_scaler:
                    X_full_scaled = StandardScaler().fit_transform(X)
                else:
                    X_full_scaled = X

                cv_scores = cross_val_score(
                    self.create_model(**model_kwargs),
                    X_full_scaled,
                    y,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error'
                )
                cv_scores = -cv_scores  # 양수로 변환
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                logger.info(f"Cross-validation RMSE: {cv_mean:.3f} ± {cv_std:.3f}")

            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")

        # 메트릭 및 결과 생성
        metrics = ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            cv_scores=cv_scores.tolist() if cv_scores is not None else None,
            cv_mean=cv_mean,
            cv_std=cv_std
        )

        result = TrainingResult(
            model=self.model,
            scaler=self.scaler,
            metrics=metrics,
            feature_names=self.feature_names,
            model_type=self.model_name,
            hyperparameters=model_kwargs,
            training_data_size=len(X_train),
            validation_data_size=len(X_val)
        )

        logger.info(
            f"Training completed - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}"
        )

        return result

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before prediction")

        X, _ = self.prepare_features(df)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_name": self.model_name,
            "is_fitted": self.is_fitted
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "BaseCommuteModel":
        """모델 로드"""
        model_data = joblib.load(filepath)

        # 모델 타입에 따라 적절한 클래스 인스턴스 생성
        model_name = model_data["model_name"]

        if model_name == "linear_regression":
            instance = LinearCommuteModel()
        elif model_name == "ridge_regression":
            instance = RidgeCommuteModel()
        elif model_name == "random_forest":
            instance = RandomForestCommuteModel()
        elif model_name == "gradient_boosting":
            instance = GradientBoostingCommuteModel()
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded from {filepath}")
        return instance


class LinearCommuteModel(BaseCommuteModel):
    """선형 회귀 모델"""

    def __init__(self):
        super().__init__("linear_regression")

    def create_model(self, **kwargs) -> BaseEstimator:
        return LinearRegression(**kwargs)


class RidgeCommuteModel(BaseCommuteModel):
    """리지 회귀 모델"""

    def __init__(self):
        super().__init__("ridge_regression")

    def create_model(self, alpha: float = 1.0, **kwargs) -> BaseEstimator:
        return Ridge(alpha=alpha, **kwargs)


class RandomForestCommuteModel(BaseCommuteModel):
    """랜덤 포레스트 모델"""

    def __init__(self):
        super().__init__("random_forest")

    def create_model(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        **kwargs
    ) -> BaseEstimator:
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )


class GradientBoostingCommuteModel(BaseCommuteModel):
    """그래디언트 부스팅 모델"""

    def __init__(self):
        super().__init__("gradient_boosting")

    def create_model(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
        **kwargs
    ) -> BaseEstimator:
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )


class ModelFactory:
    """모델 팩토리"""

    _models = {
        "linear": LinearCommuteModel,
        "ridge": RidgeCommuteModel,
        "random_forest": RandomForestCommuteModel,
        "gradient_boosting": GradientBoostingCommuteModel,
        # Enhanced 버전들 (enhanced_training.py 호환성)
        "ridge_enhanced": RidgeCommuteModel,
        "random_forest_enhanced": RandomForestCommuteModel,
        "gradient_boosting_enhanced": GradientBoostingCommuteModel,
    }

    @classmethod
    def create_model(cls, model_type: str) -> BaseCommuteModel:
        """모델 타입에 따라 모델 인스턴스 생성"""
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        return cls._models[model_type]()

    @classmethod
    def get_available_models(cls) -> List[str]:
        """사용 가능한 모델 타입 목록"""
        return list(cls._models.keys())


def compare_models(
    df: pd.DataFrame,
    model_types: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    **training_kwargs
) -> Dict[str, TrainingResult]:
    """여러 모델 성능 비교"""
    if model_types is None:
        model_types = ModelFactory.get_available_models()

    results = {}

    logger.info(f"Comparing {len(model_types)} models: {model_types}")

    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} model...")

            model = ModelFactory.create_model(model_type)
            result = model.train(
                df,
                test_size=test_size,
                random_state=random_state,
                **training_kwargs
            )
            results[model_type] = result

            logger.info(
                f"{model_type} - RMSE: {result.metrics.rmse:.3f}, "
                f"MAE: {result.metrics.mae:.3f}, R²: {result.metrics.r2:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")

    return results


def get_best_model(results: Dict[str, TrainingResult]) -> Tuple[str, TrainingResult]:
    """최적 모델 선택 (RMSE 기준)"""
    if not results:
        raise ValueError("No training results provided")

    best_model_name = min(results.keys(), key=lambda k: results[k].metrics.rmse)
    best_result = results[best_model_name]

    logger.info(f"Best model: {best_model_name} (RMSE: {best_result.metrics.rmse:.3f})")

    return best_model_name, best_result