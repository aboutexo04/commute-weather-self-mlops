"""WandB(Weights & Biases) 통합 관리자."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import wandb
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class WandBManager:
    """WandB 실험 추적 및 모니터링 관리자."""

    def __init__(
        self,
        project_name: str = "commute-weather-mlops",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        WandB 매니저 초기화.

        Args:
            project_name: WandB 프로젝트 이름
            entity: WandB 엔티티 (팀 또는 사용자명)
            api_key: WandB API 키 (환경변수에서 자동 로드)
            tags: 기본 태그 목록
        """
        self.project_name = project_name
        self.entity = entity
        self.default_tags = tags or ["mlops", "weather-prediction", "real-time"]

        # API 키 설정 (환경변수에서 자동 로드)
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        elif not os.getenv("WANDB_API_KEY"):
            # .env 파일에서 로드 시도
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            if not os.getenv("WANDB_API_KEY"):
                logger.warning("WandB API key not found. Please set WANDB_API_KEY environment variable or add it to .env file.")

        # WandB 설정
        self._configure_wandb()

        # 현재 실행 상태
        self.current_run = None
        self.run_history = []

    def _configure_wandb(self):
        """WandB 기본 설정."""
        # 오프라인 모드가 아닌 경우에만 로그인 시도
        if not os.getenv("WANDB_MODE") == "offline":
            try:
                wandb.login()
                logger.info("WandB login successful")
            except Exception as e:
                logger.warning(f"WandB login failed: {e}. Running in offline mode.")
                os.environ["WANDB_MODE"] = "offline"

    def start_run(
        self,
        run_name: Optional[str] = None,
        job_type: str = "train",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None
    ) -> wandb.run.Run:
        """새로운 WandB 실행 시작."""

        # 기존 실행이 있으면 종료
        if self.current_run is not None:
            self.finish_run()

        # 실행 이름 생성
        if not run_name:
            timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
            run_name = f"{job_type}_{timestamp}"

        # 태그 결합
        combined_tags = self.default_tags.copy()
        if tags:
            combined_tags.extend(tags)

        # 실행 시작
        self.current_run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            job_type=job_type,
            config=config,
            tags=combined_tags,
            notes=notes,
            group=group,
            reinit=True
        )

        logger.info(f"Started WandB run: {run_name}")
        return self.current_run

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        """메트릭 로깅."""
        if self.current_run is None:
            logger.warning("No active WandB run. Starting a new run.")
            self.start_run()

        # 타임스탬프 추가
        if timestamp:
            metrics["timestamp"] = timestamp.timestamp()

        wandb.log(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_model_performance(
        self,
        performance_metrics: Dict[str, float],
        model_name: str,
        model_version: str,
        step: Optional[int] = None
    ):
        """모델 성능 메트릭 로깅."""

        # 성능 메트릭에 모델 정보 추가
        metrics_with_model_info = {
            f"model_performance/{k}": v for k, v in performance_metrics.items()
        }
        metrics_with_model_info.update({
            "model_name": model_name,
            "model_version": model_version
        })

        self.log_metrics(metrics_with_model_info, step=step)

    def log_data_quality(
        self,
        data_quality_metrics: Dict[str, float],
        source: str = "asos",
        step: Optional[int] = None
    ):
        """데이터 품질 메트릭 로깅."""

        metrics = {
            f"data_quality_{source}/{k}": v for k, v in data_quality_metrics.items()
        }

        self.log_metrics(metrics, step=step)

    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        feature_set_name: str = "default"
    ):
        """피처 중요도 로깅."""

        # 피처 중요도 테이블 생성
        feature_table = wandb.Table(
            columns=["feature_name", "importance_score"],
            data=[[name, score] for name, score in zip(feature_names, importance_scores)]
        )

        # 바차트 생성
        wandb.log({
            f"feature_importance_{feature_set_name}": wandb.plot.bar(
                feature_table,
                "feature_name",
                "importance_score",
                title=f"Feature Importance - {feature_set_name}"
            )
        })

    def log_prediction_distribution(
        self,
        predictions: List[float],
        actuals: Optional[List[float]] = None,
        step: Optional[int] = None
    ):
        """예측값 분포 로깅."""

        # 예측값 히스토그램
        wandb.log({
            "predictions_histogram": wandb.Histogram(predictions)
        }, step=step)

        # 실제값이 있으면 scatter plot 생성
        if actuals:
            scatter_data = [[pred, actual] for pred, actual in zip(predictions, actuals)]
            table = wandb.Table(data=scatter_data, columns=["predictions", "actuals"])

            wandb.log({
                "predictions_vs_actuals": wandb.plot.scatter(
                    table, "predictions", "actuals",
                    title="Predictions vs Actuals"
                )
            }, step=step)

    def log_weather_data_summary(
        self,
        weather_data: Dict[str, Any],
        step: Optional[int] = None
    ):
        """날씨 데이터 요약 로깅."""

        # 기상 데이터 메트릭 추출
        weather_metrics = {}

        for key, value in weather_data.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                weather_metrics[f"weather/{key}"] = value

        # 데이터 품질 정보
        if "data_quality" in weather_data:
            weather_metrics["weather/data_quality"] = weather_data["data_quality"]

        if "feature_completeness" in weather_data:
            weather_metrics["weather/feature_completeness"] = weather_data["feature_completeness"]

        self.log_metrics(weather_metrics, step=step)

    def log_pipeline_performance(
        self,
        pipeline_metrics: Dict[str, float],
        pipeline_stage: str,
        step: Optional[int] = None
    ):
        """파이프라인 성능 메트릭 로깅."""

        metrics = {
            f"pipeline_{pipeline_stage}/{k}": v for k, v in pipeline_metrics.items()
        }

        self.log_metrics(metrics, step=step)

    def log_model_artifact(
        self,
        model_path: str,
        model_name: str,
        model_type: str = "sklearn",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """모델 아티팩트 로깅."""

        if self.current_run is None:
            logger.warning("No active WandB run for artifact logging")
            return

        # 아티팩트 생성
        artifact = wandb.Artifact(
            name=model_name,
            type=f"model_{model_type}",
            metadata=metadata or {}
        )

        # 모델 파일 추가
        artifact.add_file(model_path)

        # 아티팩트 로깅
        self.current_run.log_artifact(artifact)
        logger.info(f"Logged model artifact: {model_name}")

    def log_config_file(
        self,
        config_path: str,
        config_name: str = "pipeline_config"
    ):
        """설정 파일 로깅."""

        if self.current_run is None:
            logger.warning("No active WandB run for config logging")
            return

        artifact = wandb.Artifact(
            name=config_name,
            type="config"
        )

        artifact.add_file(config_path)
        self.current_run.log_artifact(artifact)

    def log_evaluation_report(
        self,
        evaluation_report: Dict[str, Any],
        model_name: str,
        step: Optional[int] = None
    ):
        """모델 평가 보고서 로깅."""

        # 평가 메트릭 추출
        if "performance_metrics" in evaluation_report:
            perf_metrics = evaluation_report["performance_metrics"]

            # 정확도 메트릭
            if "accuracy" in perf_metrics:
                accuracy_metrics = {
                    f"evaluation/{k}": v for k, v in perf_metrics["accuracy"].items()
                }
                self.log_metrics(accuracy_metrics, step=step)

            # 품질 메트릭
            if "quality" in perf_metrics:
                quality_metrics = {
                    f"evaluation_quality/{k}": v for k, v in perf_metrics["quality"].items()
                }
                self.log_metrics(quality_metrics, step=step)

            # 운영 메트릭
            if "operational" in perf_metrics:
                ops_metrics = {
                    f"evaluation_ops/{k}": v for k, v in perf_metrics["operational"].items()
                }
                self.log_metrics(ops_metrics, step=step)

        # 검증 결과
        if "validation_results" in evaluation_report:
            validation = evaluation_report["validation_results"]
            validation_metrics = {
                "evaluation/meets_accuracy_threshold": validation.get("meets_accuracy_threshold", False),
                "evaluation/meets_latency_threshold": validation.get("meets_latency_threshold", False),
                "evaluation/meets_stability_threshold": validation.get("meets_stability_threshold", False),
            }
            self.log_metrics(validation_metrics, step=step)

        # 전체 보고서를 JSON으로 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(evaluation_report, f, indent=2, default=str)
            temp_path = f.name

        try:
            # 보고서 아티팩트 생성
            artifact = wandb.Artifact(
                name=f"evaluation_report_{model_name}",
                type="evaluation_report"
            )
            artifact.add_file(temp_path, name="report.json")

            if self.current_run:
                self.current_run.log_artifact(artifact)

        finally:
            # 임시 파일 정리
            Path(temp_path).unlink(missing_ok=True)

    def create_alert(
        self,
        title: str,
        text: str,
        level: str = "WARN",  # INFO, WARN, ERROR
        wait_duration: int = 300  # 5분
    ):
        """WandB 알림 생성."""

        if self.current_run is None:
            logger.warning("No active WandB run for alert creation")
            return

        try:
            wandb.alert(
                title=title,
                text=text,
                level=getattr(wandb.AlertLevel, level),
                wait_duration=wait_duration
            )
            logger.info(f"Created WandB alert: {title}")
        except Exception as e:
            logger.error(f"Failed to create WandB alert: {e}")

    def watch_model(
        self,
        model,
        criterion=None,
        log: str = "gradients",
        log_freq: int = 100,
        idx: Optional[int] = None
    ):
        """모델 가중치 및 그라디언트 추적."""

        if self.current_run is None:
            logger.warning("No active WandB run for model watching")
            return

        try:
            wandb.watch(model, criterion, log=log, log_freq=log_freq, idx=idx)
            logger.info("Started watching model with WandB")
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def finish_run(self):
        """현재 실행 종료."""

        if self.current_run is not None:
            run_id = self.current_run.id
            self.run_history.append({
                "run_id": run_id,
                "finished_at": datetime.now(ZoneInfo("Asia/Seoul"))
            })

            wandb.finish()
            self.current_run = None
            logger.info(f"Finished WandB run: {run_id}")

    def get_run_summary(self) -> Optional[Dict[str, Any]]:
        """현재 실행 요약 정보 반환."""

        if self.current_run is None:
            return None

        return {
            "run_id": self.current_run.id,
            "run_name": self.current_run.name,
            "project": self.current_run.project,
            "url": self.current_run.url,
            "tags": self.current_run.tags,
            "config": dict(self.current_run.config)
        }

    def sync_offline_runs(self):
        """오프라인 실행 동기화."""

        try:
            import subprocess
            result = subprocess.run(["wandb", "sync"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Successfully synced offline WandB runs")
            else:
                logger.error(f"Failed to sync offline runs: {result.stderr}")
        except Exception as e:
            logger.error(f"Error syncing offline runs: {e}")

    @staticmethod
    def setup_sweep(
        sweep_config: Dict[str, Any],
        project_name: str,
        entity: Optional[str] = None
    ) -> str:
        """WandB Sweep 설정."""

        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=project_name,
            entity=entity
        )

        logger.info(f"Created WandB sweep: {sweep_id}")
        return sweep_id

    def __enter__(self):
        """컨텍스트 매니저 시작."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료."""
        self.finish_run()


# 전역 WandB 매니저 인스턴스
global_wandb_manager: Optional[WandBManager] = None


def get_global_wandb_manager() -> Optional[WandBManager]:
    """전역 WandB 매니저 반환."""
    return global_wandb_manager


def initialize_global_wandb_manager(
    project_name: str = "commute-weather-mlops",
    entity: Optional[str] = None,
    api_key: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> WandBManager:
    """전역 WandB 매니저 초기화."""

    global global_wandb_manager
    global_wandb_manager = WandBManager(
        project_name=project_name,
        entity=entity,
        api_key=api_key,
        tags=tags
    )

    logger.info("Initialized global WandB manager")
    return global_wandb_manager