"""통합 로깅 관리 시스템."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd

from ..storage import S3StorageManager
from ..tracking import WandBManager


class MLOpsLogFormatter(logging.Formatter):
    """MLOps 전용 로그 포매터."""

    def __init__(self):
        super().__init__()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드 포매팅."""

        # 타임스탬프 (한국 시간)
        timestamp = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Seoul"))

        # 기본 로그 정보
        log_data = {
            "timestamp": timestamp.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "process_id": os.getpid(),
            "thread_id": record.thread
        }

        # 예외 정보 추가
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # 추가 컨텍스트 정보
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        if hasattr(record, 'model_version'):
            log_data["model_version"] = record.model_version
        if hasattr(record, 'pipeline_stage'):
            log_data["pipeline_stage"] = record.pipeline_stage

        # JSON 형태로 포매팅
        import json
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class StructuredLogHandler(logging.Handler):
    """구조화된 로그 핸들러."""

    def __init__(
        self,
        s3_manager: Optional[S3StorageManager] = None,
        wandb_manager: Optional[WandBManager] = None
    ):
        super().__init__()
        self.s3_manager = s3_manager
        self.wandb_manager = wandb_manager

        # 로그 버퍼
        self.log_buffer = []
        self.buffer_size = 100

        # 에러 통계
        self.error_stats = {
            "total_errors": 0,
            "errors_by_level": {},
            "errors_by_module": {},
            "recent_errors": []
        }

    def emit(self, record: logging.LogRecord):
        """로그 레코드 처리."""
        try:
            # 로그 데이터 구조화
            log_entry = self._structure_log_record(record)

            # 버퍼에 추가
            self.log_buffer.append(log_entry)

            # 에러 통계 업데이트
            if record.levelno >= logging.ERROR:
                self._update_error_stats(record, log_entry)

            # 버퍼 크기 초과시 S3에 저장
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_to_s3()

            # 크리티컬 에러는 즉시 WandB 알림
            if record.levelno >= logging.CRITICAL:
                self._send_critical_alert(log_entry)

        except Exception as e:
            # 로깅 시스템 자체 오류는 표준 에러로 출력
            print(f"Logging error: {e}", file=sys.stderr)

    def _structure_log_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """로그 레코드를 구조화된 형태로 변환."""
        timestamp = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Seoul"))

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "level": record.levelname,
            "level_num": record.levelno,
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "message": record.getMessage(),
            "hostname": getattr(self, 'hostname', 'unknown'),
            "process_id": os.getpid(),
            "thread_id": record.thread,
            "pathname": record.pathname
        }

        # 예외 정보
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # 커스텀 속성들
        custom_attrs = ['user_id', 'request_id', 'model_version', 'pipeline_stage',
                       'prediction_id', 'data_source', 'execution_time']

        for attr in custom_attrs:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        return log_entry

    def _update_error_stats(self, record: logging.LogRecord, log_entry: Dict[str, Any]):
        """에러 통계 업데이트."""
        self.error_stats["total_errors"] += 1

        # 레벨별 통계
        level = record.levelname
        self.error_stats["errors_by_level"][level] = \
            self.error_stats["errors_by_level"].get(level, 0) + 1

        # 모듈별 통계
        module = record.module
        self.error_stats["errors_by_module"][module] = \
            self.error_stats["errors_by_module"].get(module, 0) + 1

        # 최근 에러 목록 (최대 50개)
        self.error_stats["recent_errors"].append({
            "timestamp": log_entry["timestamp"],
            "level": level,
            "module": module,
            "message": log_entry["message"]
        })

        if len(self.error_stats["recent_errors"]) > 50:
            self.error_stats["recent_errors"] = self.error_stats["recent_errors"][-50:]

    def _flush_to_s3(self):
        """로그 버퍼를 S3에 저장."""
        if not self.s3_manager or not self.log_buffer:
            return

        try:
            # DataFrame으로 변환
            df = pd.DataFrame(self.log_buffer)

            # Parquet 형식으로 직렬화
            parquet_data = df.to_parquet()

            # S3에 저장
            timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
            key = f"logs/mlops_logs_{timestamp}.parquet"

            self.s3_manager.store_raw_data(
                data=parquet_data,
                data_type="logs",
                station_id="system",
                timestamp=datetime.now(ZoneInfo("Asia/Seoul")),
                custom_key=key
            )

            # 버퍼 초기화
            self.log_buffer.clear()

        except Exception as e:
            print(f"Failed to flush logs to S3: {e}", file=sys.stderr)

    def _send_critical_alert(self, log_entry: Dict[str, Any]):
        """크리티컬 에러 알림 전송."""
        if not self.wandb_manager:
            return

        try:
            self.wandb_manager.create_alert(
                title="Critical Error Alert",
                text=f"Critical error in {log_entry['module']}.{log_entry['function']}: {log_entry['message']}",
                level="ERROR"
            )
        except Exception as e:
            print(f"Failed to send critical alert: {e}", file=sys.stderr)

    def close(self):
        """핸들러 종료 시 남은 로그 플러시."""
        if self.log_buffer:
            self._flush_to_s3()
        super().close()


class LoggingManager:
    """통합 로깅 관리자."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        s3_manager: Optional[S3StorageManager] = None,
        wandb_manager: Optional[WandBManager] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_structured: bool = True
    ):
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.s3_manager = s3_manager
        self.wandb_manager = wandb_manager

        # 로그 디렉토리 생성
        self.log_dir.mkdir(exist_ok=True)

        # 핸들러 설정
        self.handlers = {}
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_structured = enable_structured

        # 로깅 설정
        self._setup_logging()

    def _setup_logging(self):
        """로깅 시스템 설정."""
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 콘솔 핸들러
        if self.enable_console:
            self._setup_console_handler()

        # 파일 핸들러
        if self.enable_file:
            self._setup_file_handlers()

        # 구조화된 로그 핸들러
        if self.enable_structured:
            self._setup_structured_handler()

        # 외부 라이브러리 로그 레벨 조정
        self._configure_external_loggers()

    def _setup_console_handler(self):
        """콘솔 핸들러 설정."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        # 컬러풀한 포매터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        logging.getLogger().addHandler(console_handler)
        self.handlers['console'] = console_handler

    def _setup_file_handlers(self):
        """파일 핸들러 설정."""
        # 일반 로그 파일 (회전)
        log_file = self.log_dir / "mlops.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logging.getLogger().addHandler(file_handler)
        self.handlers['file'] = file_handler

        # 에러 전용 로그 파일
        error_file = self.log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        logging.getLogger().addHandler(error_handler)
        self.handlers['error'] = error_handler

    def _setup_structured_handler(self):
        """구조화된 로그 핸들러 설정."""
        structured_handler = StructuredLogHandler(
            s3_manager=self.s3_manager,
            wandb_manager=self.wandb_manager
        )
        structured_handler.setLevel(self.log_level)
        structured_handler.setFormatter(MLOpsLogFormatter())

        logging.getLogger().addHandler(structured_handler)
        self.handlers['structured'] = structured_handler

    def _configure_external_loggers(self):
        """외부 라이브러리 로거 설정."""
        # 외부 라이브러리 로그 레벨 조정
        external_loggers = {
            'urllib3': logging.WARNING,
            'requests': logging.WARNING,
            'boto3': logging.WARNING,
            'botocore': logging.WARNING,
            's3transfer': logging.WARNING,
            'mlflow': logging.WARNING,
            'wandb': logging.WARNING,
            'prefect': logging.INFO
        }

        for logger_name, level in external_loggers.items():
            logging.getLogger(logger_name).setLevel(level)

    def get_logger(self, name: str) -> logging.Logger:
        """이름별 로거 반환."""
        return logging.getLogger(name)

    def log_prediction_event(
        self,
        prediction_id: str,
        model_version: str,
        execution_time: float,
        success: bool,
        error_message: Optional[str] = None,
        **kwargs
    ):
        """예측 이벤트 로깅."""
        logger = self.get_logger("prediction")

        extra = {
            'prediction_id': prediction_id,
            'model_version': model_version,
            'execution_time': execution_time,
            'pipeline_stage': 'prediction',
            **kwargs
        }

        if success:
            logger.info(
                f"Prediction completed successfully - ID: {prediction_id}, "
                f"Time: {execution_time:.3f}s",
                extra=extra
            )
        else:
            extra['error_message'] = error_message
            logger.error(
                f"Prediction failed - ID: {prediction_id}, Error: {error_message}",
                extra=extra
            )

    def log_data_quality_event(
        self,
        data_source: str,
        quality_score: float,
        issues: List[str],
        **kwargs
    ):
        """데이터 품질 이벤트 로깅."""
        logger = self.get_logger("data_quality")

        extra = {
            'data_source': data_source,
            'quality_score': quality_score,
            'quality_issues': issues,
            'pipeline_stage': 'data_quality',
            **kwargs
        }

        if quality_score >= 80:
            logger.info(
                f"Data quality check passed - Source: {data_source}, Score: {quality_score:.1f}",
                extra=extra
            )
        elif quality_score >= 60:
            logger.warning(
                f"Data quality warning - Source: {data_source}, Score: {quality_score:.1f}, Issues: {len(issues)}",
                extra=extra
            )
        else:
            logger.error(
                f"Data quality failure - Source: {data_source}, Score: {quality_score:.1f}, Issues: {issues}",
                extra=extra
            )

    def log_model_deployment_event(
        self,
        model_name: str,
        model_version: str,
        deployment_stage: str,
        success: bool,
        **kwargs
    ):
        """모델 배포 이벤트 로깅."""
        logger = self.get_logger("deployment")

        extra = {
            'model_name': model_name,
            'model_version': model_version,
            'deployment_stage': deployment_stage,
            'pipeline_stage': 'deployment',
            **kwargs
        }

        if success:
            logger.info(
                f"Model deployment successful - {model_name} v{model_version} to {deployment_stage}",
                extra=extra
            )
        else:
            logger.error(
                f"Model deployment failed - {model_name} v{model_version} to {deployment_stage}",
                extra=extra
            )

    def log_pipeline_execution(
        self,
        pipeline_name: str,
        execution_time: float,
        success: bool,
        stages_completed: List[str],
        error_stage: Optional[str] = None,
        **kwargs
    ):
        """파이프라인 실행 로깅."""
        logger = self.get_logger("pipeline")

        extra = {
            'pipeline_name': pipeline_name,
            'execution_time': execution_time,
            'stages_completed': stages_completed,
            'pipeline_stage': 'execution',
            **kwargs
        }

        if success:
            logger.info(
                f"Pipeline executed successfully - {pipeline_name}, "
                f"Time: {execution_time:.2f}s, Stages: {len(stages_completed)}",
                extra=extra
            )
        else:
            extra['error_stage'] = error_stage
            logger.error(
                f"Pipeline execution failed - {pipeline_name}, "
                f"Failed at: {error_stage}, Completed: {len(stages_completed)}",
                extra=extra
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약 정보 반환."""
        structured_handler = self.handlers.get('structured')
        if isinstance(structured_handler, StructuredLogHandler):
            return structured_handler.error_stats.copy()
        return {"message": "Error stats not available"}

    def flush_logs(self):
        """모든 로그 버퍼 플러시."""
        for handler in self.handlers.values():
            if hasattr(handler, 'flush'):
                handler.flush()

    def close(self):
        """로깅 매니저 종료."""
        self.flush_logs()

        for handler in self.handlers.values():
            if hasattr(handler, 'close'):
                handler.close()

    def __enter__(self):
        """컨텍스트 매니저 시작."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료."""
        self.close()


# 전역 로깅 매니저 인스턴스
global_logging_manager: Optional[LoggingManager] = None


def get_global_logging_manager() -> Optional[LoggingManager]:
    """전역 로깅 매니저 반환."""
    return global_logging_manager


def initialize_global_logging_manager(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    s3_manager: Optional[S3StorageManager] = None,
    wandb_manager: Optional[WandBManager] = None,
    **kwargs
) -> LoggingManager:
    """전역 로깅 매니저 초기화."""

    global global_logging_manager
    global_logging_manager = LoggingManager(
        log_level=log_level,
        log_dir=log_dir,
        s3_manager=s3_manager,
        wandb_manager=wandb_manager,
        **kwargs
    )

    # 초기화 로그
    logger = global_logging_manager.get_logger("mlops.init")
    logger.info("MLOps Logging Manager initialized successfully")

    return global_logging_manager


def get_logger(name: str) -> logging.Logger:
    """편의 함수: 로거 반환."""
    manager = get_global_logging_manager()
    if manager:
        return manager.get_logger(name)
    return logging.getLogger(name)