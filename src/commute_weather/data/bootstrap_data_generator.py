"""초기 모델 학습을 위한 부트스트랩 데이터 생성기."""

from __future__ import annotations

import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from ..data_sources.weather_api import WeatherObservation
from ..storage.s3_manager import S3StorageManager
from ..config import ProjectPaths, KMAAPIConfig
from ..data_sources.kma_api import fetch_recent_weather_kma
from ..pipelines.baseline import compute_commute_comfort_score

logger = logging.getLogger(__name__)


@dataclass
class BootstrapConfig:
    """부트스트랩 설정"""

    # 데이터 수집 방법
    use_real_api: bool = True  # 실제 KMA API 사용 여부
    historical_days: int = 14  # API로 수집할 과거 일수

    # 합성 데이터 설정 (API 실패 시)
    synthetic_days: int = 30  # 합성 데이터 일수
    samples_per_day: int = 8  # 하루당 샘플 수 (3시간 간격)

    # 품질 기준
    min_samples_for_training: int = 100
    target_quality_score: float = 0.7

    # 시뮬레이션 설정
    seoul_weather_patterns: bool = True  # 서울 기후 패턴 반영
    seasonal_variation: bool = True  # 계절 변화 반영
    realistic_noise: bool = True  # 현실적인 노이즈 추가


class BootstrapDataGenerator:
    """초기 모델 학습용 데이터 생성기"""

    def __init__(
        self,
        storage_manager: S3StorageManager,
        kma_config: Optional[KMAAPIConfig] = None,
        config: Optional[BootstrapConfig] = None
    ):
        self.storage = storage_manager
        self.kma_config = kma_config
        self.config = config or BootstrapConfig()

        logger.info("BootstrapDataGenerator initialized")

    def generate_initial_training_data(self) -> Tuple[List[List[WeatherObservation]], List[float], Dict[str, Any]]:
        """초기 훈련 데이터 생성"""

        logger.info("🚀 Generating initial training data for cold start")

        observation_sets = []
        comfort_scores = []
        generation_stats = {
            "method": "unknown",
            "data_source": "bootstrap",
            "generation_time": datetime.now().isoformat(),
            "samples_generated": 0,
            "quality_metrics": {}
        }

        # 1. 실제 KMA API 데이터 시도
        if self.config.use_real_api and self.kma_config:
            try:
                api_data = self._collect_recent_api_data()
                if api_data:
                    observation_sets, comfort_scores = api_data
                    generation_stats["method"] = "real_api"
                    generation_stats["api_days_collected"] = self.config.historical_days
                    logger.info(f"✅ Collected {len(observation_sets)} samples from real KMA API")

            except Exception as e:
                logger.warning(f"Real API collection failed: {e}")

        # 2. 합성 데이터 생성 (API 실패 시 또는 데이터 부족 시)
        if len(observation_sets) < self.config.min_samples_for_training:
            logger.info("🧬 Generating synthetic weather data")

            needed_samples = self.config.min_samples_for_training - len(observation_sets)
            synthetic_data = self._generate_synthetic_weather_data(needed_samples)

            observation_sets.extend(synthetic_data[0])
            comfort_scores.extend(synthetic_data[1])

            generation_stats["method"] = "hybrid" if observation_sets else "synthetic"
            generation_stats["synthetic_samples"] = len(synthetic_data[0])

        # 3. 생성된 데이터 품질 평가
        quality_metrics = self._evaluate_data_quality(observation_sets, comfort_scores)
        generation_stats["quality_metrics"] = quality_metrics
        generation_stats["samples_generated"] = len(observation_sets)

        # 4. S3에 부트스트랩 데이터 저장
        self._save_bootstrap_data_to_s3(observation_sets, comfort_scores, generation_stats)

        logger.info(f"✅ Bootstrap data generation completed: {len(observation_sets)} samples")
        logger.info(f"📊 Quality score: {quality_metrics.get('overall_quality', 0):.3f}")

        return observation_sets, comfort_scores, generation_stats

    def _collect_recent_api_data(self) -> Optional[Tuple[List[List[WeatherObservation]], List[float]]]:
        """실제 KMA API에서 최근 데이터 수집"""

        logger.info(f"Collecting recent {self.config.historical_days} days from KMA API")

        try:
            all_observations = []

            # 최근 몇 일간의 데이터를 3시간 단위로 수집
            for days_ago in range(self.config.historical_days):
                try:
                    # 각 날짜마다 3시간씩 수집
                    observations = fetch_recent_weather_kma(
                        self.kma_config,
                        lookback_hours=3
                    )

                    if observations:
                        all_observations.append(observations)

                    # API 호출 제한을 위해 약간의 지연
                    import time
                    time.sleep(1)

                except Exception as e:
                    logger.warning(f"Failed to collect data for {days_ago} days ago: {e}")
                    continue

            if len(all_observations) >= 10:  # 최소 10개 샘플
                # 편안함 점수 계산
                comfort_scores = []
                for obs_set in all_observations:
                    try:
                        comfort = compute_commute_comfort_score(obs_set)
                        comfort_scores.append(comfort.score)
                    except:
                        comfort_scores.append(0.5)  # 기본값

                return all_observations, comfort_scores

            return None

        except Exception as e:
            logger.error(f"API data collection failed: {e}")
            return None

    def _generate_synthetic_weather_data(self, target_samples: int) -> Tuple[List[List[WeatherObservation]], List[float]]:
        """합성 날씨 데이터 생성"""

        logger.info(f"Generating {target_samples} synthetic weather samples")

        observation_sets = []
        comfort_scores = []

        # 현재 시기 (계절성 반영)
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        current_month = now.month

        # 서울 기후 패턴
        climate_patterns = self._get_seoul_climate_patterns()
        current_climate = climate_patterns[current_month - 1]

        # 샘플 생성
        for _ in range(target_samples):
            # 시간 분산 (지난 30일간)
            days_back = np.random.uniform(0, 30)
            sample_time = now - timedelta(days=days_back)

            # 3시간 윈도우 생성
            observations = self._generate_weather_window(sample_time, current_climate)

            # 편안함 점수 계산
            try:
                comfort = compute_commute_comfort_score(observations)
                comfort_score = comfort.score
            except:
                comfort_score = self._calculate_synthetic_comfort_score(observations)

            observation_sets.append(observations)
            comfort_scores.append(comfort_score)

        logger.info(f"Generated {len(observation_sets)} synthetic weather samples")
        return observation_sets, comfort_scores

    def _get_seoul_climate_patterns(self) -> List[Dict[str, Any]]:
        """서울 월별 기후 패턴"""

        return [
            # 1월
            {"temp_range": (-5, 5), "humidity_range": (40, 70), "wind_range": (1, 6), "precip_prob": 0.1},
            # 2월
            {"temp_range": (-2, 8), "humidity_range": (35, 65), "wind_range": (1, 7), "precip_prob": 0.1},
            # 3월
            {"temp_range": (3, 15), "humidity_range": (40, 70), "wind_range": (2, 8), "precip_prob": 0.15},
            # 4월
            {"temp_range": (8, 20), "humidity_range": (45, 75), "wind_range": (2, 7), "precip_prob": 0.2},
            # 5월
            {"temp_range": (15, 25), "humidity_range": (50, 80), "wind_range": (1, 6), "precip_prob": 0.25},
            # 6월
            {"temp_range": (20, 28), "humidity_range": (65, 85), "wind_range": (1, 5), "precip_prob": 0.4},
            # 7월
            {"temp_range": (24, 32), "humidity_range": (70, 90), "wind_range": (1, 4), "precip_prob": 0.5},
            # 8월
            {"temp_range": (25, 33), "humidity_range": (70, 90), "wind_range": (1, 5), "precip_prob": 0.45},
            # 9월
            {"temp_range": (20, 28), "humidity_range": (60, 80), "wind_range": (1, 6), "precip_prob": 0.3},
            # 10월
            {"temp_range": (12, 22), "humidity_range": (50, 75), "wind_range": (2, 7), "precip_prob": 0.2},
            # 11월
            {"temp_range": (5, 15), "humidity_range": (45, 70), "wind_range": (2, 8), "precip_prob": 0.15},
            # 12월
            {"temp_range": (-2, 8), "humidity_range": (40, 65), "wind_range": (2, 7), "precip_prob": 0.1},
        ]

    def _generate_weather_window(self, base_time: datetime, climate: Dict[str, Any]) -> List[WeatherObservation]:
        """3시간 날씨 윈도우 생성"""

        observations = []

        # 기본 날씨 조건 생성
        base_temp = np.random.uniform(*climate["temp_range"])
        base_humidity = np.random.uniform(*climate["humidity_range"])
        base_wind = np.random.uniform(*climate["wind_range"])

        # 강수 여부 결정
        has_precipitation = np.random.random() < climate["precip_prob"]
        base_precip = np.random.uniform(0.1, 5.0) if has_precipitation else 0.0

        # 3시간 동안의 관측값 생성
        for hour_offset in range(3):
            timestamp = base_time + timedelta(hours=hour_offset)

            # 시간에 따른 자연스러운 변화
            temp_variation = np.random.normal(0, 1)  # ±1도 변화
            humidity_variation = np.random.normal(0, 3)  # ±3% 변화
            wind_variation = np.random.normal(0, 0.5)  # ±0.5m/s 변화

            temperature = base_temp + temp_variation
            humidity = max(10, min(95, base_humidity + humidity_variation))
            wind_speed = max(0, base_wind + wind_variation)

            # 강수량 (지속적이거나 점진적 변화)
            if has_precipitation:
                precip_variation = np.random.uniform(0.8, 1.2)  # ±20% 변화
                precipitation = base_precip * precip_variation
            else:
                precipitation = 0.0

            # 강수 타입 결정
            if precipitation > 0:
                precip_type = "snow" if temperature <= 2 else "rain"
            else:
                precip_type = "none"

            observation = WeatherObservation(
                timestamp=timestamp,
                temperature_c=round(temperature, 1),
                wind_speed_ms=round(wind_speed, 1),
                precipitation_mm=round(precipitation, 1),
                relative_humidity=round(humidity, 1),
                precipitation_type=precip_type
            )

            observations.append(observation)

        return observations

    def _calculate_synthetic_comfort_score(self, observations: List[WeatherObservation]) -> float:
        """합성 데이터용 편안함 점수 계산"""

        if not observations:
            return 0.5

        latest = observations[-1]

        # 온도 점수 (18-24°C 최적)
        temp = latest.temperature_c
        if 18 <= temp <= 24:
            temp_score = 1.0
        elif 10 <= temp <= 30:
            temp_score = 0.7 - abs(temp - 21) * 0.02
        else:
            temp_score = max(0.1, 0.5 - abs(temp - 21) * 0.03)

        # 습도 점수 (40-60% 최적)
        humidity = latest.relative_humidity or 50
        if 40 <= humidity <= 60:
            humidity_score = 1.0
        elif 30 <= humidity <= 80:
            humidity_score = 0.8 - abs(humidity - 50) * 0.01
        else:
            humidity_score = max(0.1, 0.6 - abs(humidity - 50) * 0.02)

        # 바람 점수 (0-5m/s 적당)
        wind = latest.wind_speed_ms
        if wind <= 5:
            wind_score = 1.0 - wind * 0.1
        elif wind <= 10:
            wind_score = 0.5 - (wind - 5) * 0.05
        else:
            wind_score = max(0.1, 0.25 - (wind - 10) * 0.02)

        # 강수 점수
        precip = latest.precipitation_mm
        if precip == 0:
            precip_score = 1.0
        elif precip <= 1:
            precip_score = 0.8
        elif precip <= 5:
            precip_score = 0.5
        else:
            precip_score = 0.2

        # 종합 점수 (가중 평균)
        comfort_score = (
            temp_score * 0.35 +
            humidity_score * 0.25 +
            wind_score * 0.2 +
            precip_score * 0.2
        )

        # 노이즈 추가 (현실적인 변동성)
        if self.config.realistic_noise:
            noise = np.random.normal(0, 0.05)  # ±5% 노이즈
            comfort_score += noise

        return max(0.0, min(1.0, comfort_score))

    def _evaluate_data_quality(
        self,
        observation_sets: List[List[WeatherObservation]],
        comfort_scores: List[float]
    ) -> Dict[str, Any]:
        """생성된 데이터 품질 평가"""

        if not observation_sets:
            return {"overall_quality": 0.0, "issues": ["No data generated"]}

        quality_metrics = {
            "total_samples": len(observation_sets),
            "total_observations": sum(len(obs_set) for obs_set in observation_sets),
            "issues": [],
            "strengths": []
        }

        # 1. 샘플 수 확인
        if len(observation_sets) >= self.config.min_samples_for_training:
            quality_metrics["strengths"].append(f"Sufficient samples: {len(observation_sets)}")
        else:
            quality_metrics["issues"].append(f"Insufficient samples: {len(observation_sets)}")

        # 2. 타겟 분포 확인
        comfort_std = np.std(comfort_scores)
        comfort_range = max(comfort_scores) - min(comfort_scores)

        if comfort_std > 0.15:
            quality_metrics["strengths"].append(f"Good target variance: {comfort_std:.3f}")
        else:
            quality_metrics["issues"].append(f"Low target variance: {comfort_std:.3f}")

        if comfort_range > 0.4:
            quality_metrics["strengths"].append(f"Good target range: {comfort_range:.3f}")

        # 3. 데이터 다양성 확인
        all_temps = []
        all_humidities = []
        all_winds = []

        for obs_set in observation_sets:
            for obs in obs_set:
                all_temps.append(obs.temperature_c)
                if obs.relative_humidity:
                    all_humidities.append(obs.relative_humidity)
                all_winds.append(obs.wind_speed_ms)

        temp_range = max(all_temps) - min(all_temps)
        humidity_range = max(all_humidities) - min(all_humidities) if all_humidities else 0
        wind_range = max(all_winds) - min(all_winds)

        diversity_score = 0
        if temp_range > 20:  # 20도 이상 온도 범위
            diversity_score += 1
        if humidity_range > 30:  # 30% 이상 습도 범위
            diversity_score += 1
        if wind_range > 5:  # 5m/s 이상 바람 범위
            diversity_score += 1

        quality_metrics["diversity_score"] = diversity_score / 3

        # 4. 전체 품질 점수 계산
        quality_factors = []

        # 샘플 수 점수
        sample_score = min(1.0, len(observation_sets) / self.config.min_samples_for_training)
        quality_factors.append(sample_score)

        # 다양성 점수
        quality_factors.append(quality_metrics["diversity_score"])

        # 타겟 분포 점수
        target_score = min(1.0, comfort_std / 0.2)  # 0.2 표준편차가 이상적
        quality_factors.append(target_score)

        overall_quality = np.mean(quality_factors)
        quality_metrics["overall_quality"] = overall_quality

        # 추가 통계
        quality_metrics.update({
            "comfort_score_stats": {
                "mean": float(np.mean(comfort_scores)),
                "std": float(np.std(comfort_scores)),
                "min": float(min(comfort_scores)),
                "max": float(max(comfort_scores))
            },
            "weather_stats": {
                "temperature_range": f"{min(all_temps):.1f}°C - {max(all_temps):.1f}°C",
                "humidity_range": f"{min(all_humidities):.1f}% - {max(all_humidities):.1f}%" if all_humidities else "N/A",
                "wind_range": f"{min(all_winds):.1f} - {max(all_winds):.1f} m/s"
            }
        })

        return quality_metrics

    def _save_bootstrap_data_to_s3(
        self,
        observation_sets: List[List[WeatherObservation]],
        comfort_scores: List[float],
        generation_stats: Dict[str, Any]
    ) -> None:
        """부트스트랩 데이터를 S3에 저장"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 부트스트랩 메타데이터 저장
            metadata_path = f"bootstrap_data/metadata/bootstrap_{timestamp}.json"
            self.storage.write_json(metadata_path, generation_stats)

            # 부트스트랩 데이터 요약 저장
            summary = {
                "generation_time": generation_stats["generation_time"],
                "method": generation_stats["method"],
                "samples_count": len(observation_sets),
                "quality_score": generation_stats["quality_metrics"].get("overall_quality", 0),
                "comfort_score_range": [min(comfort_scores), max(comfort_scores)] if comfort_scores else [0, 0]
            }

            summary_path = f"bootstrap_data/summaries/summary_{timestamp}.json"
            self.storage.write_json(summary_path, summary)

            logger.info(f"Bootstrap data saved to S3: {len(observation_sets)} samples")

        except Exception as e:
            logger.warning(f"Failed to save bootstrap data to S3: {e}")


def create_bootstrap_generator(
    s3_bucket: str,
    kma_auth_key: Optional[str] = None,
    project_root: Optional[Path] = None,
    config: Optional[BootstrapConfig] = None
) -> BootstrapDataGenerator:
    """부트스트랩 데이터 생성기 생성"""

    if project_root is None:
        project_root = Path.cwd()

    paths = ProjectPaths.from_root(project_root)
    storage = S3StorageManager(paths=paths, bucket=s3_bucket)

    # KMA 설정 (API 키가 있는 경우)
    kma_config = None
    if kma_auth_key:
        kma_config = KMAAPIConfig(
            base_url="https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php",
            auth_key=kma_auth_key,
            station_id="108"  # 서울
        )

    return BootstrapDataGenerator(
        storage_manager=storage,
        kma_config=kma_config,
        config=config
    )