"""지능형 피처 엔지니어링 시스템."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from .base import BaseFeatureTransformer, FeatureMetadata
from ..data_sources.asos_collector import AsosDataCollector

logger = logging.getLogger(__name__)


class IntelligentFeatureSelector:
    """지능형 피처 선택기"""

    def __init__(self):
        self.feature_importance_history: Dict[str, List[float]] = {}
        self.correlation_threshold = 0.8
        self.importance_threshold = 0.1

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        max_features: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """지능형 피처 선택"""

        if df.empty:
            return [], {}

        # 기본 피처 중요도 계산
        feature_scores = self._calculate_base_importance(df, target_column)

        # 상관관계 기반 중복 피처 제거
        selected_features = self._remove_redundant_features(df, feature_scores)

        # 최대 피처 수 제한
        if max_features and len(selected_features) > max_features:
            # 중요도 순으로 정렬하여 상위 N개 선택
            sorted_features = sorted(
                selected_features,
                key=lambda f: feature_scores.get(f, 0),
                reverse=True
            )
            selected_features = sorted_features[:max_features]

        # 선택된 피처들의 점수만 반환
        selected_scores = {f: feature_scores[f] for f in selected_features}

        logger.info(f"Selected {len(selected_features)} features from {len(df.columns)} candidates")

        return selected_features, selected_scores

    def _calculate_base_importance(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, float]:
        """기본 피처 중요도 계산"""

        importance_scores = {}

        # 숫자형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)

        for col in numeric_columns:
            try:
                # 기본 통계 기반 중요도
                series = df[col].dropna()

                if len(series) == 0:
                    importance_scores[col] = 0.0
                    continue

                # 분산 기반 점수 (분산이 낮으면 정보가 적음)
                variance_score = min(1.0, series.var() / (series.mean() + 1e-6)) if series.mean() != 0 else 0.0

                # 비결측값 비율
                completeness_score = len(series) / len(df)

                # 유니크 값 비율 (너무 많으면 노이즈, 너무 적으면 정보 부족)
                unique_ratio = len(series.unique()) / len(series)
                uniqueness_score = min(1.0, unique_ratio * 2) if unique_ratio < 0.5 else 1.0

                # 목표 변수와의 상관관계 (있는 경우)
                correlation_score = 0.5  # 기본값

                if target_column and target_column in df.columns:
                    try:
                        corr = abs(df[col].corr(df[target_column]))
                        if not pd.isna(corr):
                            correlation_score = corr
                    except Exception:
                        pass

                # 종합 점수 계산
                final_score = (
                    variance_score * 0.3 +
                    completeness_score * 0.3 +
                    uniqueness_score * 0.2 +
                    correlation_score * 0.2
                )

                importance_scores[col] = final_score

            except Exception as e:
                logger.warning(f"Failed to calculate importance for {col}: {e}")
                importance_scores[col] = 0.0

        return importance_scores

    def _remove_redundant_features(
        self,
        df: pd.DataFrame,
        feature_scores: Dict[str, float]
    ) -> List[str]:
        """상관관계가 높은 중복 피처 제거"""

        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr().abs()

        # 높은 상관관계를 가진 피처 쌍 찾기
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]

                if corr_value > self.correlation_threshold:
                    high_corr_pairs.append((col1, col2, corr_value))

        # 중복 피처 제거 (중요도가 낮은 것을 제거)
        features_to_remove = set()

        for col1, col2, corr_value in high_corr_pairs:
            score1 = feature_scores.get(col1, 0)
            score2 = feature_scores.get(col2, 0)

            # 중요도가 낮은 피처를 제거 대상으로 선정
            if score1 < score2:
                features_to_remove.add(col1)
            else:
                features_to_remove.add(col2)

        # 최종 선택된 피처 목록
        selected_features = [
            col for col in feature_scores.keys()
            if col not in features_to_remove and feature_scores[col] > self.importance_threshold
        ]

        if high_corr_pairs:
            logger.info(f"Removed {len(features_to_remove)} redundant features due to high correlation")

        return selected_features


class RealTimeFeatureEngineer(BaseFeatureTransformer):
    """실시간 피처 엔지니어링"""

    def __init__(self, version: str = "2.0.0"):
        super().__init__("realtime_feature_engineer", version)

        self.feature_selector = IntelligentFeatureSelector()
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.feature_metadata = [
            FeatureMetadata(
                name="temp_anomaly_score",
                description="온도 이상치 점수 (0-1)",
                feature_type="derived",
                source_columns=["temperature_c"],
                creation_logic="Isolation Forest anomaly detection",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 1.0),
                nullable=False
            ),
            FeatureMetadata(
                name="weather_pattern_index",
                description="날씨 패턴 인덱스 (PCA 기반)",
                feature_type="derived",
                source_columns=["temperature_c", "relative_humidity_pct", "wind_speed_ms"],
                creation_logic="First principal component of weather variables",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(-3.0, 3.0),
                nullable=False
            ),
            FeatureMetadata(
                name="comfort_probability",
                description="쾌적 확률 (ML 기반 예측)",
                feature_type="derived",
                source_columns=["temperature_c", "relative_humidity_pct", "precipitation_mm_1h", "wind_speed_ms"],
                creation_logic="Logistic regression on historical comfort data",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 1.0),
                nullable=False
            ),
            FeatureMetadata(
                name="feature_confidence",
                description="피처 신뢰도 점수",
                feature_type="derived",
                source_columns=["data_quality", "feature_completeness"],
                creation_logic="Weighted combination of quality metrics",
                version=version,
                created_at=datetime.now(),
                data_type="float",
                expected_range=(0.0, 1.0),
                nullable=False
            ),
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """실시간 피처 변환"""

        if df.empty:
            return df

        result_df = df.copy()

        # 1. 핵심 피처 선택
        selected_features, feature_scores = self.feature_selector.select_features(
            df, max_features=10
        )

        logger.info(f"Selected features: {selected_features}")

        # 2. 이상치 탐지
        result_df = self._add_anomaly_features(result_df, selected_features)

        # 3. 패턴 추출
        result_df = self._add_pattern_features(result_df, selected_features)

        # 4. 쾌적도 확률 계산
        result_df = self._add_comfort_probability(result_df)

        # 5. 피처 신뢰도 계산
        result_df = self._add_feature_confidence(result_df)

        # 6. 동적 피처 정규화
        result_df = self._normalize_features(result_df, selected_features)

        return result_df

    def _add_anomaly_features(
        self,
        df: pd.DataFrame,
        selected_features: List[str]
    ) -> pd.DataFrame:
        """이상치 기반 피처 추가"""

        result_df = df.copy()

        # 온도 이상치 점수
        if "temperature_c" in df.columns and not df["temperature_c"].isna().all():
            temp_values = df["temperature_c"].dropna().values.reshape(-1, 1)

            if len(temp_values) > 1:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_scores = iso_forest.fit_predict(temp_values)

                    # 점수를 0-1 범위로 변환 (-1: 이상치, 1: 정상)
                    normalized_scores = (anomaly_scores + 1) / 2
                    result_df["temp_anomaly_score"] = normalized_scores[0] if len(normalized_scores) == 1 else np.mean(normalized_scores)

                except Exception as e:
                    logger.warning(f"Failed to calculate temperature anomaly: {e}")
                    result_df["temp_anomaly_score"] = 0.5
            else:
                result_df["temp_anomaly_score"] = 0.5
        else:
            result_df["temp_anomaly_score"] = 0.0

        return result_df

    def _add_pattern_features(
        self,
        df: pd.DataFrame,
        selected_features: List[str]
    ) -> pd.DataFrame:
        """패턴 기반 피처 추가"""

        result_df = df.copy()

        # 기상 패턴 인덱스 (PCA 기반)
        weather_columns = ["temperature_c", "relative_humidity_pct", "wind_speed_ms"]
        available_weather_cols = [col for col in weather_columns if col in df.columns]

        if len(available_weather_cols) >= 2:
            try:
                weather_data = df[available_weather_cols].dropna()

                if len(weather_data) > 0:
                    # PCA로 첫 번째 주성분 추출
                    pca = PCA(n_components=1, random_state=42)
                    pattern_index = pca.fit_transform(weather_data.values)

                    result_df["weather_pattern_index"] = pattern_index[0, 0] if len(pattern_index) > 0 else 0.0
                else:
                    result_df["weather_pattern_index"] = 0.0

            except Exception as e:
                logger.warning(f"Failed to calculate weather pattern index: {e}")
                result_df["weather_pattern_index"] = 0.0
        else:
            result_df["weather_pattern_index"] = 0.0

        return result_df

    def _add_comfort_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """쾌적도 확률 계산"""

        result_df = df.copy()

        # 간단한 규칙 기반 쾌적도 확률 (나중에 ML 모델로 대체 가능)
        comfort_score = 1.0

        # 온도 기반 확률
        if "temperature_c" in df.columns and not df["temperature_c"].isna().all():
            temp = df["temperature_c"].mean()
            if 15 <= temp <= 25:
                temp_prob = 1.0
            elif 10 <= temp < 15 or 25 < temp <= 30:
                temp_prob = 0.7
            elif 5 <= temp < 10 or 30 < temp <= 35:
                temp_prob = 0.4
            else:
                temp_prob = 0.1

            comfort_score *= temp_prob

        # 강수 기반 확률
        if "precipitation_mm_1h" in df.columns and not df["precipitation_mm_1h"].isna().all():
            precip = df["precipitation_mm_1h"].sum()
            if precip == 0:
                precip_prob = 1.0
            elif precip <= 1:
                precip_prob = 0.8
            elif precip <= 5:
                precip_prob = 0.5
            else:
                precip_prob = 0.2

            comfort_score *= precip_prob

        # 풍속 기반 확률
        if "wind_speed_ms" in df.columns and not df["wind_speed_ms"].isna().all():
            wind = df["wind_speed_ms"].max()
            if wind <= 3:
                wind_prob = 1.0
            elif wind <= 7:
                wind_prob = 0.8
            elif wind <= 12:
                wind_prob = 0.5
            else:
                wind_prob = 0.2

            comfort_score *= wind_prob

        result_df["comfort_probability"] = max(0.0, min(1.0, comfort_score))

        return result_df

    def _add_feature_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 신뢰도 계산"""

        result_df = df.copy()

        confidence_score = 1.0

        # 데이터 품질 기반 신뢰도
        if "data_quality" in df.columns:
            quality = df["data_quality"].mean() / 100.0
            confidence_score *= quality

        # 피처 완성도 기반 신뢰도
        if "feature_completeness" in df.columns:
            completeness = df["feature_completeness"].mean() / 100.0
            confidence_score *= completeness

        # 시간 신선도 기반 신뢰도
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            latest_time = pd.to_datetime(df["timestamp"]).max()
            current_time = datetime.now()

            if pd.notna(latest_time):
                time_diff = current_time - latest_time
                hours_diff = time_diff.total_seconds() / 3600

                if hours_diff <= 1:
                    time_confidence = 1.0
                elif hours_diff <= 3:
                    time_confidence = 0.8
                elif hours_diff <= 6:
                    time_confidence = 0.6
                else:
                    time_confidence = 0.3

                confidence_score *= time_confidence

        result_df["feature_confidence"] = max(0.0, min(1.0, confidence_score))

        return result_df

    def _normalize_features(
        self,
        df: pd.DataFrame,
        selected_features: List[str]
    ) -> pd.DataFrame:
        """동적 피처 정규화"""

        result_df = df.copy()

        # 정규화할 피처 선별 (숫자형이면서 선택된 피처)
        normalize_candidates = [
            col for col in selected_features
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        for col in normalize_candidates:
            try:
                values = df[col].dropna()

                if len(values) > 0:
                    # MinMax 정규화 (0-1 범위)
                    min_val = values.min()
                    max_val = values.max()

                    if max_val != min_val:
                        normalized_col = f"{col}_normalized"
                        result_df[normalized_col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        result_df[f"{col}_normalized"] = 0.5  # 상수값인 경우 중간값

            except Exception as e:
                logger.warning(f"Failed to normalize {col}: {e}")

        return result_df

    def get_feature_names(self) -> List[str]:
        """생성되는 피처 이름 목록"""
        return [
            "temp_anomaly_score",
            "weather_pattern_index",
            "comfort_probability",
            "feature_confidence"
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """피팅과 변환을 동시에 수행"""
        self.is_fitted = True
        return self.transform(df)


class AdaptiveFeaturePipeline:
    """적응형 피처 파이프라인"""

    def __init__(self, asos_collector: AsosDataCollector):
        self.asos_collector = asos_collector
        self.feature_engineer = RealTimeFeatureEngineer()
        self.feature_history: List[Dict[str, Any]] = []
        self.performance_feedback: Dict[str, float] = {}

    def process_realtime_data(
        self,
        station_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """실시간 데이터 처리"""

        # 1. 최신 ASOS 데이터 수집
        observation = self.asos_collector.fetch_latest_observation(station_id)

        if not observation:
            logger.warning("Failed to fetch latest observation")
            return None

        # 2. 핵심 피처 추출
        core_features = self.asos_collector.select_core_features(observation)

        # 3. DataFrame으로 변환
        feature_df = pd.DataFrame([core_features])

        # 4. 지능형 피처 엔지니어링 적용
        engineered_df = self.feature_engineer.fit_transform(feature_df)

        # 5. 결과 준비
        result = engineered_df.to_dict(orient='records')[0]

        # 6. 피처 요약 정보 추가
        feature_summary = self.asos_collector.create_feature_summary(core_features)
        result.update(feature_summary)

        # 7. 이력 저장
        self.feature_history.append(result)

        # 최근 100개 이력만 유지
        if len(self.feature_history) > 100:
            self.feature_history = self.feature_history[-100:]

        logger.info(
            f"Processed realtime data: quality={result.get('data_quality', 0):.1f}, "
            f"confidence={result.get('feature_confidence', 0):.1f}"
        )

        return result

    def get_feature_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """피처 트렌드 분석"""

        if len(self.feature_history) < 2:
            return {"status": "insufficient_data", "hours_back": hours_back}

        recent_data = self.feature_history[-hours_back:] if len(self.feature_history) >= hours_back else self.feature_history

        trends = {}

        # 주요 피처들의 트렌드 계산
        key_features = [
            "temperature_c", "relative_humidity_pct", "precipitation_mm_1h",
            "wind_speed_ms", "comfort_probability", "data_quality"
        ]

        for feature in key_features:
            values = [item.get(feature) for item in recent_data if item.get(feature) is not None]

            if len(values) >= 2:
                # 간단한 트렌드 계산 (첫 값 대비 마지막 값의 변화율)
                trend_change = (values[-1] - values[0]) / abs(values[0] + 1e-6) * 100
                trends[f"{feature}_trend"] = trend_change

                # 안정성 계산 (표준편차)
                trends[f"{feature}_stability"] = np.std(values)

        trends["data_points"] = len(recent_data)
        trends["time_span_hours"] = hours_back

        return trends

    def update_performance_feedback(
        self,
        actual_comfort_score: float,
        predicted_comfort_probability: float
    ) -> None:
        """성능 피드백 업데이트"""

        # 예측 정확도 계산
        prediction_error = abs(actual_comfort_score/100.0 - predicted_comfort_probability)

        # 피드백 이력 저장
        timestamp = datetime.now().isoformat()
        self.performance_feedback[timestamp] = {
            "actual": actual_comfort_score,
            "predicted": predicted_comfort_probability,
            "error": prediction_error
        }

        # 최근 50개 피드백만 유지
        if len(self.performance_feedback) > 50:
            sorted_keys = sorted(self.performance_feedback.keys())
            for old_key in sorted_keys[:-50]:
                del self.performance_feedback[old_key]

        logger.info(f"Updated performance feedback: error={prediction_error:.3f}")

    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 계산"""

        if not self.performance_feedback:
            return {"status": "no_feedback_data"}

        errors = [fb["error"] for fb in self.performance_feedback.values()]

        return {
            "mean_absolute_error": np.mean(errors),
            "std_error": np.std(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "feedback_count": len(errors),
            "accuracy_score": max(0.0, 1.0 - np.mean(errors))  # 1 - MAE
        }