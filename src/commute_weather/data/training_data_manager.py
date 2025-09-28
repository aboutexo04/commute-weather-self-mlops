"""실시간 수집 데이터를 활용한 훈련 데이터 관리자."""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from ..data_sources.weather_api import WeatherObservation
from ..data_sources.kma_api import _parse_typ01_response
from ..storage.s3_manager import S3StorageManager
from ..config import ProjectPaths
from ..pipelines.baseline import compute_commute_comfort_score

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataStats:
    """훈련 데이터 통계"""
    total_observations: int
    valid_observation_sets: int
    date_range_start: datetime
    date_range_end: datetime
    average_quality_score: float
    comfort_score_range: Tuple[float, float]
    data_completeness: float


class RealTimeTrainingDataManager:
    """실시간 수집 데이터 기반 훈련 데이터 관리"""

    def __init__(
        self,
        storage_manager: S3StorageManager,
        min_quality_score: float = 0.6,
        min_observations_per_set: int = 3
    ):
        self.storage = storage_manager
        self.min_quality_score = min_quality_score
        self.min_observations_per_set = min_observations_per_set

        logger.info("RealTimeTrainingDataManager initialized")

    def collect_training_data(
        self,
        days_back: int = 30,
        station_id: str = "108"
    ) -> Tuple[List[List[WeatherObservation]], List[float], TrainingDataStats]:
        """실시간 수집된 데이터로부터 훈련 데이터 구성"""

        logger.info(f"Collecting training data from last {days_back} days")

        end_date = datetime.now(ZoneInfo("Asia/Seoul"))
        start_date = end_date - timedelta(days=days_back)

        observation_sets = []
        comfort_scores = []
        quality_scores = []
        total_observations = 0

        # 날짜별로 데이터 수집
        current_date = start_date
        while current_date <= end_date:
            try:
                daily_data = self._load_daily_observations(current_date, station_id)

                if daily_data:
                    # 3시간 윈도우로 그룹화
                    hourly_groups = self._group_observations_by_window(daily_data)

                    for window_obs in hourly_groups:
                        if len(window_obs) >= self.min_observations_per_set:
                            # 데이터 품질 확인
                            quality_score = self._calculate_data_quality(window_obs)

                            if quality_score >= self.min_quality_score:
                                # 편안함 점수 계산
                                comfort = compute_commute_comfort_score(window_obs)

                                observation_sets.append(window_obs)
                                comfort_scores.append(comfort.score)
                                quality_scores.append(quality_score)
                                total_observations += len(window_obs)

            except Exception as e:
                logger.warning(f"Failed to process data for {current_date.date()}: {e}")

            current_date += timedelta(days=1)

        # 통계 계산
        stats = self._calculate_training_stats(
            observation_sets, comfort_scores, quality_scores,
            total_observations, start_date, end_date
        )

        logger.info(f"Collected {len(observation_sets)} training samples")
        logger.info(f"Quality score range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
        logger.info(f"Comfort score range: {min(comfort_scores):.3f} - {max(comfort_scores):.3f}")

        return observation_sets, comfort_scores, stats

    def _load_daily_observations(
        self,
        date: datetime,
        station_id: str
    ) -> List[WeatherObservation]:
        """특정 날짜의 관측 데이터 로드"""

        try:
            # S3에서 해당 날짜의 원시 데이터 파일들 찾기
            date_prefix = f"raw_data/kma/{station_id}/{date.strftime('%Y/%m/%d')}"
            files = self.storage.list_files_by_prefix(date_prefix)

            all_observations = []

            for file_path in files:
                try:
                    # 원시 KMA 데이터 읽기
                    raw_content = self.storage.read_text(file_path)

                    # KMA 형식 파싱
                    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                    day_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

                    observations = _parse_typ01_response(
                        raw_content,
                        start_time=day_start,
                        end_time=day_end
                    )

                    all_observations.extend(observations)

                except Exception as e:
                    logger.warning(f"Failed to parse file {file_path}: {e}")
                    continue

            # 시간순 정렬 및 중복 제거
            if all_observations:
                # 타임스탬프 기준 정렬
                all_observations.sort(key=lambda x: x.timestamp)

                # 중복 타임스탬프 제거 (동일 시간의 경우 마지막 것만 유지)
                unique_observations = []
                seen_timestamps = set()

                for obs in all_observations:
                    timestamp_key = obs.timestamp.replace(second=0, microsecond=0)
                    if timestamp_key not in seen_timestamps:
                        unique_observations.append(obs)
                        seen_timestamps.add(timestamp_key)

                return unique_observations

            return []

        except Exception as e:
            logger.error(f"Failed to load daily observations for {date.date()}: {e}")
            return []

    def _group_observations_by_window(
        self,
        observations: List[WeatherObservation],
        window_hours: int = 3
    ) -> List[List[WeatherObservation]]:
        """관측 데이터를 시간 윈도우별로 그룹화"""

        if not observations:
            return []

        groups = []
        window_size = timedelta(hours=window_hours)

        # 시작 시간을 정시로 정규화
        start_time = observations[0].timestamp.replace(minute=0, second=0, microsecond=0)
        end_time = observations[-1].timestamp

        current_window_start = start_time

        while current_window_start < end_time:
            window_end = current_window_start + window_size

            # 현재 윈도우에 속하는 관측값들
            window_observations = [
                obs for obs in observations
                if current_window_start <= obs.timestamp < window_end
            ]

            if len(window_observations) >= self.min_observations_per_set:
                groups.append(window_observations)

            current_window_start += timedelta(hours=1)  # 1시간씩 슬라이딩

        return groups

    def _calculate_data_quality(self, observations: List[WeatherObservation]) -> float:
        """관측 데이터의 품질 점수 계산"""

        if not observations:
            return 0.0

        quality_factors = []

        # 1. 데이터 완성도
        total_fields = len(observations) * 5  # 5개 주요 필드
        missing_fields = 0

        for obs in observations:
            if obs.temperature_c == 0.0:  # 의심스러운 0도
                missing_fields += 1
            if obs.relative_humidity is None:
                missing_fields += 1
            if obs.wind_speed_ms < 0:
                missing_fields += 1
            if obs.precipitation_mm < 0:
                missing_fields += 1

        completeness = 1.0 - (missing_fields / total_fields)
        quality_factors.append(completeness)

        # 2. 값의 합리성
        reasonableness = 1.0
        for obs in observations:
            # 극값 확인
            if abs(obs.temperature_c) > 50:  # 극한 온도
                reasonableness -= 0.1
            if obs.wind_speed_ms > 50:  # 극한 풍속
                reasonableness -= 0.1
            if obs.precipitation_mm > 100:  # 극한 강수량
                reasonableness -= 0.1

        quality_factors.append(max(0, reasonableness))

        # 3. 시계열 일관성
        if len(observations) > 1:
            temp_changes = []
            for i in range(1, len(observations)):
                temp_diff = abs(observations[i].temperature_c - observations[i-1].temperature_c)
                temp_changes.append(temp_diff)

            # 급격한 온도 변화 확인 (1시간에 10도 이상 변화는 비정상)
            extreme_changes = sum(1 for change in temp_changes if change > 10)
            consistency = 1.0 - (extreme_changes / len(temp_changes))
            quality_factors.append(consistency)

        # 4. 시간 간격 일관성
        if len(observations) > 1:
            time_gaps = []
            for i in range(1, len(observations)):
                gap = (observations[i].timestamp - observations[i-1].timestamp).total_seconds() / 3600
                time_gaps.append(gap)

            # 이상적인 간격은 1시간
            gap_consistency = 1.0 - min(1.0, np.std(time_gaps) / 2.0)
            quality_factors.append(gap_consistency)

        # 종합 품질 점수 (가중 평균)
        overall_quality = np.mean(quality_factors)
        return float(overall_quality)

    def _calculate_training_stats(
        self,
        observation_sets: List[List[WeatherObservation]],
        comfort_scores: List[float],
        quality_scores: List[float],
        total_observations: int,
        start_date: datetime,
        end_date: datetime
    ) -> TrainingDataStats:
        """훈련 데이터 통계 계산"""

        if not observation_sets:
            return TrainingDataStats(
                total_observations=0,
                valid_observation_sets=0,
                date_range_start=start_date,
                date_range_end=end_date,
                average_quality_score=0.0,
                comfort_score_range=(0.0, 0.0),
                data_completeness=0.0
            )

        # 모든 타임스탬프 수집
        all_timestamps = []
        for obs_set in observation_sets:
            all_timestamps.extend([obs.timestamp for obs in obs_set])

        actual_start = min(all_timestamps) if all_timestamps else start_date
        actual_end = max(all_timestamps) if all_timestamps else end_date

        # 데이터 완성도 계산
        expected_hours = (end_date - start_date).total_seconds() / 3600
        actual_hours = len(observation_sets) * 3  # 각 세트당 평균 3시간
        completeness = min(1.0, actual_hours / expected_hours)

        return TrainingDataStats(
            total_observations=total_observations,
            valid_observation_sets=len(observation_sets),
            date_range_start=actual_start,
            date_range_end=actual_end,
            average_quality_score=float(np.mean(quality_scores)) if quality_scores else 0.0,
            comfort_score_range=(float(min(comfort_scores)), float(max(comfort_scores))) if comfort_scores else (0.0, 0.0),
            data_completeness=float(completeness)
        )

    def get_recent_observations_for_prediction(
        self,
        hours_back: int = 3,
        station_id: str = "108"
    ) -> List[WeatherObservation]:
        """예측용 최근 관측 데이터 가져오기"""

        logger.info(f"Getting recent observations for prediction ({hours_back} hours)")

        try:
            end_time = datetime.now(ZoneInfo("Asia/Seoul"))
            start_time = end_time - timedelta(hours=hours_back)

            # 최근 데이터 파일들 검색
            all_observations = []

            # 최근 몇 일의 데이터를 확인
            for days_back in range(3):  # 최근 3일간 확인
                check_date = end_time - timedelta(days=days_back)
                daily_obs = self._load_daily_observations(check_date, station_id)

                # 지정된 시간 범위 내의 관측값만 필터링
                filtered_obs = [
                    obs for obs in daily_obs
                    if start_time <= obs.timestamp <= end_time
                ]

                all_observations.extend(filtered_obs)

            # 시간순 정렬
            all_observations.sort(key=lambda x: x.timestamp)

            # 최근 관측값들만 반환 (중복 제거)
            if all_observations:
                # 마지막 몇 개 관측값 반환
                recent_count = min(hours_back, len(all_observations))
                return all_observations[-recent_count:]

            return []

        except Exception as e:
            logger.error(f"Failed to get recent observations: {e}")
            return []

    def create_training_dataset_summary(
        self,
        observation_sets: List[List[WeatherObservation]],
        comfort_scores: List[float],
        stats: TrainingDataStats
    ) -> Dict[str, Any]:
        """훈련 데이터셋 요약 생성"""

        summary = {
            "dataset_info": {
                "total_samples": len(observation_sets),
                "total_observations": stats.total_observations,
                "date_range": {
                    "start": stats.date_range_start.isoformat(),
                    "end": stats.date_range_end.isoformat(),
                    "duration_days": (stats.date_range_end - stats.date_range_start).days
                },
                "data_quality": {
                    "average_quality_score": stats.average_quality_score,
                    "data_completeness": stats.data_completeness,
                    "quality_threshold": self.min_quality_score
                }
            },
            "target_distribution": {
                "comfort_score_range": stats.comfort_score_range,
                "comfort_score_mean": float(np.mean(comfort_scores)),
                "comfort_score_std": float(np.std(comfort_scores)),
                "comfort_score_samples": len(comfort_scores)
            },
            "data_source": {
                "collection_method": "real_time_kma_api",
                "storage_backend": "s3",
                "processing_version": "enhanced_v1",
                "feature_engineering": "kma_42_features"
            }
        }

        return summary


def create_training_data_manager(
    s3_bucket: str,
    project_root: Optional[Path] = None,
    min_quality_score: float = 0.6
) -> RealTimeTrainingDataManager:
    """훈련 데이터 매니저 생성"""

    if project_root is None:
        project_root = Path.cwd()

    paths = ProjectPaths.from_root(project_root)
    storage = S3StorageManager(paths=paths, bucket=s3_bucket)

    return RealTimeTrainingDataManager(
        storage_manager=storage,
        min_quality_score=min_quality_score
    )