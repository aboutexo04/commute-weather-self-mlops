"""데이터 품질 검증 모듈."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """검증 결과 심각도 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """데이터 검증 결과"""
    name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class DataQualityValidator:
    """기상 데이터 품질 검증기"""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def reset_results(self) -> None:
        """검증 결과 초기화"""
        self.results = []

    def validate_weather_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """기상 데이터 전체 검증"""
        self.reset_results()

        # 기본 구조 검증
        self._validate_basic_structure(df)

        # 데이터 타입 검증
        self._validate_data_types(df)

        # 값 범위 검증
        self._validate_value_ranges(df)

        # 시계열 연속성 검증
        self._validate_temporal_continuity(df)

        # 누락 데이터 검증
        self._validate_missing_data(df)

        # 이상치 검증
        self._validate_outliers(df)

        # 데이터 일관성 검증
        self._validate_data_consistency(df)

        return self.results

    def _add_result(
        self,
        name: str,
        severity: ValidationSeverity,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """검증 결과 추가"""
        result = ValidationResult(
            name=name,
            severity=severity,
            passed=passed,
            message=message,
            details=details
        )
        self.results.append(result)

        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL,
        }[severity]

        logger.log(log_level, f"[{name}] {message}")

    def _validate_basic_structure(self, df: pd.DataFrame) -> None:
        """기본 데이터 구조 검증"""
        required_columns = [
            "timestamp",
            "temperature_c",
            "wind_speed_ms",
            "precipitation_mm",
            "relative_humidity"
        ]

        # 데이터프레임이 비어있는지 확인
        if df.empty:
            self._add_result(
                "empty_dataframe",
                ValidationSeverity.CRITICAL,
                False,
                "DataFrame is empty",
                {"row_count": 0}
            )
            return

        # 필수 컬럼 존재 확인
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self._add_result(
                "missing_columns",
                ValidationSeverity.CRITICAL,
                False,
                f"Missing required columns: {missing_columns}",
                {"missing_columns": missing_columns, "available_columns": list(df.columns)}
            )
        else:
            self._add_result(
                "required_columns",
                ValidationSeverity.INFO,
                True,
                "All required columns present",
                {"column_count": len(df.columns)}
            )

        # 최소 데이터 행 수 확인
        min_rows = 1
        if len(df) < min_rows:
            self._add_result(
                "insufficient_rows",
                ValidationSeverity.ERROR,
                False,
                f"Insufficient data rows: {len(df)} < {min_rows}",
                {"row_count": len(df), "min_required": min_rows}
            )
        else:
            self._add_result(
                "sufficient_rows",
                ValidationSeverity.INFO,
                True,
                f"Sufficient data rows: {len(df)}",
                {"row_count": len(df)}
            )

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """데이터 타입 검증"""
        if df.empty:
            return

        # 타임스탬프 검증
        if "timestamp" in df.columns:
            try:
                pd.to_datetime(df["timestamp"])
                self._add_result(
                    "timestamp_format",
                    ValidationSeverity.INFO,
                    True,
                    "Timestamp format is valid"
                )
            except (ValueError, TypeError) as e:
                self._add_result(
                    "timestamp_format",
                    ValidationSeverity.ERROR,
                    False,
                    f"Invalid timestamp format: {e}"
                )

        # 숫자형 컬럼 검증
        numeric_columns = ["temperature_c", "wind_speed_ms", "precipitation_mm", "relative_humidity"]
        for col in numeric_columns:
            if col in df.columns:
                non_numeric_count = pd.to_numeric(df[col], errors='coerce').isna().sum() - df[col].isna().sum()
                if non_numeric_count > 0:
                    self._add_result(
                        f"{col}_numeric",
                        ValidationSeverity.WARNING,
                        False,
                        f"Non-numeric values found in {col}: {non_numeric_count} values",
                        {"non_numeric_count": int(non_numeric_count), "column": col}
                    )
                else:
                    self._add_result(
                        f"{col}_numeric",
                        ValidationSeverity.INFO,
                        True,
                        f"All values in {col} are numeric"
                    )

    def _validate_value_ranges(self, df: pd.DataFrame) -> None:
        """값 범위 검증"""
        if df.empty:
            return

        # 온도 범위 검증 (-50°C ~ 60°C)
        if "temperature_c" in df.columns:
            temp_series = pd.to_numeric(df["temperature_c"], errors='coerce')
            out_of_range = ((temp_series < -50) | (temp_series > 60)).sum()

            if out_of_range > 0:
                self._add_result(
                    "temperature_range",
                    ValidationSeverity.WARNING,
                    False,
                    f"Temperature values out of valid range (-50°C to 60°C): {out_of_range} values",
                    {
                        "out_of_range_count": int(out_of_range),
                        "min_value": float(temp_series.min()) if not temp_series.empty else None,
                        "max_value": float(temp_series.max()) if not temp_series.empty else None
                    }
                )
            else:
                self._add_result(
                    "temperature_range",
                    ValidationSeverity.INFO,
                    True,
                    "All temperature values within valid range"
                )

        # 풍속 범위 검증 (0 ~ 100 m/s)
        if "wind_speed_ms" in df.columns:
            wind_series = pd.to_numeric(df["wind_speed_ms"], errors='coerce')
            out_of_range = ((wind_series < 0) | (wind_series > 100)).sum()

            if out_of_range > 0:
                self._add_result(
                    "wind_speed_range",
                    ValidationSeverity.WARNING,
                    False,
                    f"Wind speed values out of valid range (0-100 m/s): {out_of_range} values",
                    {"out_of_range_count": int(out_of_range)}
                )
            else:
                self._add_result(
                    "wind_speed_range",
                    ValidationSeverity.INFO,
                    True,
                    "All wind speed values within valid range"
                )

        # 강수량 범위 검증 (0 ~ 1000 mm)
        if "precipitation_mm" in df.columns:
            precip_series = pd.to_numeric(df["precipitation_mm"], errors='coerce')
            out_of_range = ((precip_series < 0) | (precip_series > 1000)).sum()

            if out_of_range > 0:
                self._add_result(
                    "precipitation_range",
                    ValidationSeverity.WARNING,
                    False,
                    f"Precipitation values out of valid range (0-1000 mm): {out_of_range} values",
                    {"out_of_range_count": int(out_of_range)}
                )
            else:
                self._add_result(
                    "precipitation_range",
                    ValidationSeverity.INFO,
                    True,
                    "All precipitation values within valid range"
                )

        # 습도 범위 검증 (0 ~ 100%)
        if "relative_humidity" in df.columns:
            humidity_series = pd.to_numeric(df["relative_humidity"], errors='coerce')
            out_of_range = ((humidity_series < 0) | (humidity_series > 100)).sum()

            if out_of_range > 0:
                self._add_result(
                    "humidity_range",
                    ValidationSeverity.WARNING,
                    False,
                    f"Humidity values out of valid range (0-100%): {out_of_range} values",
                    {"out_of_range_count": int(out_of_range)}
                )
            else:
                self._add_result(
                    "humidity_range",
                    ValidationSeverity.INFO,
                    True,
                    "All humidity values within valid range"
                )

    def _validate_temporal_continuity(self, df: pd.DataFrame) -> None:
        """시계열 연속성 검증"""
        if df.empty or "timestamp" not in df.columns:
            return

        try:
            timestamps = pd.to_datetime(df["timestamp"]).sort_values()

            if len(timestamps) < 2:
                self._add_result(
                    "temporal_continuity",
                    ValidationSeverity.INFO,
                    True,
                    "Single timestamp - continuity check not applicable"
                )
                return

            # 시간 간격 계산
            time_diffs = timestamps.diff().dropna()

            # 예상 간격 (1시간)
            expected_interval = timedelta(hours=1)
            tolerance = timedelta(minutes=10)  # 10분 허용오차

            # 불규칙한 간격 찾기
            irregular_intervals = time_diffs[
                (time_diffs < expected_interval - tolerance) |
                (time_diffs > expected_interval + tolerance)
            ]

            if len(irregular_intervals) > 0:
                self._add_result(
                    "temporal_continuity",
                    ValidationSeverity.WARNING,
                    False,
                    f"Irregular time intervals detected: {len(irregular_intervals)} gaps",
                    {
                        "irregular_count": len(irregular_intervals),
                        "expected_interval_hours": 1,
                        "max_gap_hours": float(time_diffs.max().total_seconds() / 3600),
                        "min_gap_hours": float(time_diffs.min().total_seconds() / 3600)
                    }
                )
            else:
                self._add_result(
                    "temporal_continuity",
                    ValidationSeverity.INFO,
                    True,
                    "Time intervals are regular"
                )

        except Exception as e:
            self._add_result(
                "temporal_continuity",
                ValidationSeverity.ERROR,
                False,
                f"Error validating temporal continuity: {e}"
            )

    def _validate_missing_data(self, df: pd.DataFrame) -> None:
        """누락 데이터 검증"""
        if df.empty:
            return

        critical_columns = ["timestamp", "temperature_c"]
        important_columns = ["wind_speed_ms", "precipitation_mm"]
        optional_columns = ["relative_humidity"]

        # 중요 컬럼의 누락 데이터 확인
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / len(df) * 100

                if missing_count > 0:
                    self._add_result(
                        f"missing_{col}",
                        ValidationSeverity.CRITICAL,
                        False,
                        f"Missing values in critical column {col}: {missing_count} ({missing_ratio:.1f}%)",
                        {"missing_count": int(missing_count), "missing_ratio": float(missing_ratio)}
                    )
                else:
                    self._add_result(
                        f"missing_{col}",
                        ValidationSeverity.INFO,
                        True,
                        f"No missing values in critical column {col}"
                    )

        # 일반 컬럼의 누락 데이터 확인
        for col in important_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / len(df) * 100

                if missing_ratio > 20:  # 20% 이상 누락
                    self._add_result(
                        f"missing_{col}",
                        ValidationSeverity.WARNING,
                        False,
                        f"High missing values in {col}: {missing_count} ({missing_ratio:.1f}%)",
                        {"missing_count": int(missing_count), "missing_ratio": float(missing_ratio)}
                    )
                elif missing_count > 0:
                    self._add_result(
                        f"missing_{col}",
                        ValidationSeverity.INFO,
                        True,
                        f"Acceptable missing values in {col}: {missing_count} ({missing_ratio:.1f}%)",
                        {"missing_count": int(missing_count), "missing_ratio": float(missing_ratio)}
                    )
                else:
                    self._add_result(
                        f"missing_{col}",
                        ValidationSeverity.INFO,
                        True,
                        f"No missing values in {col}"
                    )

    def _validate_outliers(self, df: pd.DataFrame) -> None:
        """이상치 검증 (IQR 방법 사용)"""
        if df.empty:
            return

        numeric_columns = ["temperature_c", "wind_speed_ms", "precipitation_mm", "relative_humidity"]

        for col in numeric_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()

                if len(series) < 4:  # IQR 계산을 위한 최소 데이터 필요
                    continue

                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_ratio = len(outliers) / len(series) * 100

                if outlier_ratio > 10:  # 10% 이상이 이상치
                    self._add_result(
                        f"outliers_{col}",
                        ValidationSeverity.WARNING,
                        False,
                        f"High outlier ratio in {col}: {len(outliers)} ({outlier_ratio:.1f}%)",
                        {
                            "outlier_count": len(outliers),
                            "outlier_ratio": float(outlier_ratio),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound)
                        }
                    )
                elif len(outliers) > 0:
                    self._add_result(
                        f"outliers_{col}",
                        ValidationSeverity.INFO,
                        True,
                        f"Acceptable outlier ratio in {col}: {len(outliers)} ({outlier_ratio:.1f}%)",
                        {"outlier_count": len(outliers), "outlier_ratio": float(outlier_ratio)}
                    )
                else:
                    self._add_result(
                        f"outliers_{col}",
                        ValidationSeverity.INFO,
                        True,
                        f"No outliers detected in {col}"
                    )

    def _validate_data_consistency(self, df: pd.DataFrame) -> None:
        """데이터 일관성 검증"""
        if df.empty:
            return

        # 습도와 강수량의 일관성 확인
        if "relative_humidity" in df.columns and "precipitation_mm" in df.columns:
            humidity = pd.to_numeric(df["relative_humidity"], errors='coerce')
            precipitation = pd.to_numeric(df["precipitation_mm"], errors='coerce')

            # 강수가 있는데 습도가 낮은 경우 (60% 미만)
            inconsistent = ((precipitation > 0) & (humidity < 60)).sum()

            if inconsistent > 0:
                inconsistent_ratio = inconsistent / len(df) * 100
                self._add_result(
                    "humidity_precipitation_consistency",
                    ValidationSeverity.WARNING,
                    False,
                    f"Inconsistent humidity-precipitation relationships: {inconsistent} cases ({inconsistent_ratio:.1f}%)",
                    {"inconsistent_count": int(inconsistent), "inconsistent_ratio": float(inconsistent_ratio)}
                )
            else:
                self._add_result(
                    "humidity_precipitation_consistency",
                    ValidationSeverity.INFO,
                    True,
                    "Humidity-precipitation relationships are consistent"
                )

        # 중복 타임스탬프 확인
        if "timestamp" in df.columns:
            try:
                timestamps = pd.to_datetime(df["timestamp"])
                duplicate_count = timestamps.duplicated().sum()

                if duplicate_count > 0:
                    self._add_result(
                        "duplicate_timestamps",
                        ValidationSeverity.ERROR,
                        False,
                        f"Duplicate timestamps found: {duplicate_count}",
                        {"duplicate_count": int(duplicate_count)}
                    )
                else:
                    self._add_result(
                        "duplicate_timestamps",
                        ValidationSeverity.INFO,
                        True,
                        "No duplicate timestamps"
                    )
            except Exception:
                # 타임스탬프 파싱 실패는 이미 다른 검증에서 확인됨
                pass

    def get_summary(self) -> Dict[str, Any]:
        """검증 결과 요약"""
        summary = {
            "total_checks": len(self.results),
            "passed_checks": sum(1 for r in self.results if r.passed),
            "failed_checks": sum(1 for r in self.results if not r.passed),
            "severity_counts": {
                "info": sum(1 for r in self.results if r.severity == ValidationSeverity.INFO),
                "warning": sum(1 for r in self.results if r.severity == ValidationSeverity.WARNING),
                "error": sum(1 for r in self.results if r.severity == ValidationSeverity.ERROR),
                "critical": sum(1 for r in self.results if r.severity == ValidationSeverity.CRITICAL),
            },
            "overall_quality": "good"  # 기본값
        }

        # 전체 품질 평가
        critical_failures = summary["severity_counts"]["critical"]
        error_failures = summary["severity_counts"]["error"]
        warning_failures = summary["severity_counts"]["warning"]

        if critical_failures > 0:
            summary["overall_quality"] = "critical"
        elif error_failures > 0:
            summary["overall_quality"] = "poor"
        elif warning_failures > len(self.results) * 0.3:  # 30% 이상이 경고
            summary["overall_quality"] = "fair"
        else:
            summary["overall_quality"] = "good"

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """검증 결과를 딕셔너리로 변환"""
        return {
            "summary": self.get_summary(),
            "results": [
                {
                    "name": r.name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None
                }
                for r in self.results
            ]
        }