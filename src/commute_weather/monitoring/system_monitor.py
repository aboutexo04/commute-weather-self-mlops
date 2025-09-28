"""시스템 모니터링 및 알림 시스템."""

from __future__ import annotations

import logging
import os
import psutil
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from ..tracking import WandBManager
from ..storage import S3StorageManager

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭 정보."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int

    # 애플리케이션 특화 메트릭
    prediction_count: int = 0
    prediction_latency: float = 0.0
    error_count: int = 0
    model_version: Optional[str] = None


@dataclass
class AlertRule:
    """알림 규칙 정의."""
    name: str
    metric: str  # cpu_usage, memory_usage, error_rate 등
    threshold: float
    operator: str  # >, <, >=, <=, ==
    duration_minutes: int  # 지속 시간
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message_template: str

    # 알림 설정
    enabled: bool = True
    cooldown_minutes: int = 60  # 재알림 방지 시간

    def evaluate(self, value: float) -> bool:
        """임계값 평가."""
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return value == self.threshold
        return False


class SystemMonitor:
    """시스템 모니터링 관리자."""

    def __init__(
        self,
        s3_manager: Optional[S3StorageManager] = None,
        wandb_manager: Optional[WandBManager] = None,
        monitoring_interval: int = 60  # 초
    ):
        self.s3_manager = s3_manager
        self.wandb_manager = wandb_manager
        self.monitoring_interval = monitoring_interval

        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_thread = None

        # 메트릭 저장소
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1440  # 24시간 (1분마다 수집)

        # 알림 규칙
        self.alert_rules = self._create_default_alert_rules()
        self.active_alerts: Dict[str, datetime] = {}

        # 애플리케이션 메트릭 카운터
        self.app_metrics = {
            "prediction_count": 0,
            "error_count": 0,
            "total_latency": 0.0,
            "prediction_times": []
        }

        # 메트릭 수집 콜백
        self.metric_collectors: List[Callable[[], Dict[str, Any]]] = []

    def _create_default_alert_rules(self) -> List[AlertRule]:
        """기본 알림 규칙 생성."""
        return [
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_usage",
                threshold=80.0,
                operator=">=",
                duration_minutes=5,
                severity="WARNING",
                message_template="CPU 사용률이 {value:.1f}%로 높습니다.",
                cooldown_minutes=30
            ),
            AlertRule(
                name="high_memory_usage",
                metric="memory_usage",
                threshold=85.0,
                operator=">=",
                duration_minutes=3,
                severity="WARNING",
                message_template="메모리 사용률이 {value:.1f}%로 높습니다.",
                cooldown_minutes=30
            ),
            AlertRule(
                name="disk_space_low",
                metric="disk_usage",
                threshold=90.0,
                operator=">=",
                duration_minutes=1,
                severity="ERROR",
                message_template="디스크 사용률이 {value:.1f}%로 매우 높습니다.",
                cooldown_minutes=60
            ),
            AlertRule(
                name="high_error_rate",
                metric="error_rate",
                threshold=5.0,
                operator=">=",
                duration_minutes=2,
                severity="ERROR",
                message_template="에러율이 {value:.1f}%로 높습니다.",
                cooldown_minutes=15
            ),
            AlertRule(
                name="prediction_latency_high",
                metric="prediction_latency",
                threshold=2000.0,  # 2초
                operator=">=",
                duration_minutes=3,
                severity="WARNING",
                message_template="예측 응답시간이 {value:.0f}ms로 높습니다.",
                cooldown_minutes=20
            )
        ]

    def start_monitoring(self):
        """모니터링 시작."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("System monitoring started")

    def stop_monitoring(self):
        """모니터링 중지."""
        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """모니터링 루프."""
        while self.is_monitoring:
            try:
                # 시스템 메트릭 수집
                metrics = self._collect_system_metrics()

                # 애플리케이션 메트릭 추가
                self._add_application_metrics(metrics)

                # 커스텀 메트릭 수집
                self._collect_custom_metrics(metrics)

                # 메트릭 저장
                self._store_metrics(metrics)

                # 알림 규칙 평가
                self._evaluate_alert_rules(metrics)

                # WandB 로깅
                if self.wandb_manager:
                    self._log_to_wandb(metrics)

                # 메트릭 히스토리 관리
                self._cleanup_old_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집."""
        # CPU 사용률
        cpu_usage = psutil.cpu_percent(interval=1)

        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100

        # 네트워크 I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }

        # 프로세스 수
        process_count = len(psutil.pids())

        return SystemMetrics(
            timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count
        )

    def _add_application_metrics(self, metrics: SystemMetrics):
        """애플리케이션 메트릭 추가."""
        # 예측 횟수
        metrics.prediction_count = self.app_metrics["prediction_count"]

        # 평균 응답시간 계산
        if self.app_metrics["prediction_times"]:
            metrics.prediction_latency = sum(self.app_metrics["prediction_times"]) / len(self.app_metrics["prediction_times"])

        # 에러 횟수
        metrics.error_count = self.app_metrics["error_count"]

    def _collect_custom_metrics(self, metrics: SystemMetrics):
        """커스텀 메트릭 수집."""
        for collector in self.metric_collectors:
            try:
                custom_metrics = collector()
                # 커스텀 메트릭을 메인 메트릭에 추가
                for key, value in custom_metrics.items():
                    setattr(metrics, key, value)
            except Exception as e:
                logger.error(f"Error collecting custom metrics: {e}")

    def _store_metrics(self, metrics: SystemMetrics):
        """메트릭 저장."""
        # 메모리에 저장
        self.metrics_history.append(metrics)

        # S3에 주기적으로 저장 (매 시간)
        if self.s3_manager and len(self.metrics_history) % 60 == 0:
            self._backup_metrics_to_s3()

    def _backup_metrics_to_s3(self):
        """메트릭을 S3에 백업."""
        try:
            # 최근 1시간 메트릭 추출
            recent_metrics = self.metrics_history[-60:]

            # DataFrame으로 변환
            data = []
            for metric in recent_metrics:
                row = {
                    "timestamp": metric.timestamp.isoformat(),
                    "cpu_usage": metric.cpu_usage,
                    "memory_usage": metric.memory_usage,
                    "disk_usage": metric.disk_usage,
                    "process_count": metric.process_count,
                    "prediction_count": metric.prediction_count,
                    "prediction_latency": metric.prediction_latency,
                    "error_count": metric.error_count,
                    "model_version": metric.model_version
                }
                data.append(row)

            df = pd.DataFrame(data)

            # S3에 저장
            timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
            key = f"monitoring/system_metrics_{timestamp}.parquet"

            # parquet 형식으로 저장
            parquet_data = df.to_parquet()

            self.s3_manager.store_raw_data(
                data=parquet_data,
                data_type="monitoring",
                station_id="system",
                timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
                custom_key=key
            )

            logger.info(f"Backed up {len(recent_metrics)} metrics to S3: {key}")

        except Exception as e:
            logger.error(f"Failed to backup metrics to S3: {e}")

    def _evaluate_alert_rules(self, metrics: SystemMetrics):
        """알림 규칙 평가."""
        current_time = datetime.now(ZoneInfo("Asia/Seoul"))

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # 메트릭 값 가져오기
            metric_value = self._get_metric_value(metrics, rule.metric)
            if metric_value is None:
                continue

            # 임계값 평가
            if rule.evaluate(metric_value):
                # 쿨다운 시간 확인
                last_alert_time = self.active_alerts.get(rule.name)
                if last_alert_time:
                    cooldown_end = last_alert_time + timedelta(minutes=rule.cooldown_minutes)
                    if current_time < cooldown_end:
                        continue  # 쿨다운 중이므로 알림 건너뛰기

                # 지속 시간 확인 (최근 N분간 지속적으로 임계값 초과)
                if self._check_duration_threshold(rule, metrics):
                    self._trigger_alert(rule, metric_value, current_time)
                    self.active_alerts[rule.name] = current_time

    def _get_metric_value(self, metrics: SystemMetrics, metric_name: str) -> Optional[float]:
        """메트릭 값 추출."""
        if metric_name == "cpu_usage":
            return metrics.cpu_usage
        elif metric_name == "memory_usage":
            return metrics.memory_usage
        elif metric_name == "disk_usage":
            return metrics.disk_usage
        elif metric_name == "prediction_latency":
            return metrics.prediction_latency
        elif metric_name == "error_rate":
            # 최근 1시간 에러율 계산
            return self._calculate_error_rate()
        elif hasattr(metrics, metric_name):
            return getattr(metrics, metric_name)
        return None

    def _calculate_error_rate(self) -> float:
        """에러율 계산."""
        if not self.metrics_history:
            return 0.0

        # 최근 1시간 메트릭
        one_hour_ago = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= one_hour_ago
        ]

        if not recent_metrics:
            return 0.0

        total_predictions = sum(m.prediction_count for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)

        if total_predictions == 0:
            return 0.0

        return (total_errors / total_predictions) * 100

    def _check_duration_threshold(self, rule: AlertRule, current_metrics: SystemMetrics) -> bool:
        """지속 시간 임계값 확인."""
        if rule.duration_minutes <= 0:
            return True  # 즉시 알림

        # 필요한 데이터 포인트 수 계산
        required_points = rule.duration_minutes

        if len(self.metrics_history) < required_points:
            return False

        # 최근 N분간의 메트릭 확인
        recent_metrics = self.metrics_history[-required_points:]

        for metrics in recent_metrics:
            metric_value = self._get_metric_value(metrics, rule.metric)
            if metric_value is None or not rule.evaluate(metric_value):
                return False

        return True

    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """알림 트리거."""
        message = rule.message_template.format(value=value)

        logger.log(
            logging.ERROR if rule.severity in ["ERROR", "CRITICAL"] else logging.WARNING,
            f"ALERT [{rule.severity}] {rule.name}: {message}"
        )

        # WandB 알림
        if self.wandb_manager:
            wandb_level = "ERROR" if rule.severity in ["ERROR", "CRITICAL"] else "WARN"
            self.wandb_manager.create_alert(
                title=f"System Alert: {rule.name}",
                text=f"{message} (임계값: {rule.threshold}, 현재값: {value:.2f})",
                level=wandb_level
            )

    def _log_to_wandb(self, metrics: SystemMetrics):
        """WandB에 메트릭 로깅."""
        try:
            wandb_metrics = {
                "system/cpu_usage": metrics.cpu_usage,
                "system/memory_usage": metrics.memory_usage,
                "system/disk_usage": metrics.disk_usage,
                "system/process_count": metrics.process_count,
                "system/prediction_count": metrics.prediction_count,
                "system/prediction_latency": metrics.prediction_latency,
                "system/error_count": metrics.error_count,
                "system/error_rate": self._calculate_error_rate()
            }

            # 네트워크 메트릭
            for key, value in metrics.network_io.items():
                wandb_metrics[f"system/network_{key}"] = value

            self.wandb_manager.log_metrics(wandb_metrics)

        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")

    def _cleanup_old_metrics(self):
        """오래된 메트릭 정리."""
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

    def add_metric_collector(self, collector: Callable[[], Dict[str, Any]]):
        """커스텀 메트릭 수집기 추가."""
        self.metric_collectors.append(collector)

    def record_prediction(self, latency_ms: float, success: bool = True):
        """예측 이벤트 기록."""
        self.app_metrics["prediction_count"] += 1
        self.app_metrics["prediction_times"].append(latency_ms)

        # 최근 100개 응답시간만 유지
        if len(self.app_metrics["prediction_times"]) > 100:
            self.app_metrics["prediction_times"] = self.app_metrics["prediction_times"][-100:]

        if not success:
            self.app_metrics["error_count"] += 1

    def record_error(self, error_type: str = "general"):
        """에러 이벤트 기록."""
        self.app_metrics["error_count"] += 1

        # WandB에 즉시 로깅
        if self.wandb_manager:
            self.wandb_manager.log_metrics({
                f"errors/{error_type}": 1.0,
                "errors/total": self.app_metrics["error_count"]
            })

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """현재 메트릭 반환."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """메트릭 요약 정보 반환."""
        cutoff_time = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"message": "No metrics available"}

        # 평균, 최대, 최소값 계산
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        disk_values = [m.disk_usage for m in recent_metrics]
        latency_values = [m.prediction_latency for m in recent_metrics if m.prediction_latency > 0]

        summary = {
            "time_range_hours": hours,
            "data_points": len(recent_metrics),
            "cpu_usage": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_usage": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "disk_usage": {
                "avg": sum(disk_values) / len(disk_values),
                "max": max(disk_values),
                "min": min(disk_values)
            },
            "total_predictions": sum(m.prediction_count for m in recent_metrics),
            "total_errors": sum(m.error_count for m in recent_metrics),
            "error_rate": self._calculate_error_rate()
        }

        if latency_values:
            summary["prediction_latency"] = {
                "avg": sum(latency_values) / len(latency_values),
                "max": max(latency_values),
                "min": min(latency_values)
            }

        return summary

    def add_alert_rule(self, rule: AlertRule):
        """커스텀 알림 규칙 추가."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def update_alert_rule(self, rule_name: str, **kwargs):
        """알림 규칙 업데이트."""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"Updated alert rule: {rule_name}")
                return True
        return False

    def disable_alert_rule(self, rule_name: str):
        """알림 규칙 비활성화."""
        return self.update_alert_rule(rule_name, enabled=False)

    def enable_alert_rule(self, rule_name: str):
        """알림 규칙 활성화."""
        return self.update_alert_rule(rule_name, enabled=True)


# 전역 시스템 모니터 인스턴스
global_system_monitor: Optional[SystemMonitor] = None


def get_global_system_monitor() -> Optional[SystemMonitor]:
    """전역 시스템 모니터 반환."""
    return global_system_monitor


def initialize_global_system_monitor(
    s3_manager: Optional[S3StorageManager] = None,
    wandb_manager: Optional[WandBManager] = None,
    monitoring_interval: int = 60
) -> SystemMonitor:
    """전역 시스템 모니터 초기화."""

    global global_system_monitor
    global_system_monitor = SystemMonitor(
        s3_manager=s3_manager,
        wandb_manager=wandb_manager,
        monitoring_interval=monitoring_interval
    )

    logger.info("Initialized global system monitor")
    return global_system_monitor