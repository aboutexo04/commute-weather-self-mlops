"""S3-based storage manager for MLOps data pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3StorageManager:
    """
    S3 기반 스토리지 매니저
    MLOps 파이프라인을 위한 계층적 데이터 구조 관리
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "ap-northeast-2",
    ):
        """
        S3 스토리지 매니저 초기화

        Args:
            bucket_name: S3 버킷 이름
            prefix: 모든 키에 적용할 접두사
            aws_access_key_id: AWS 액세스 키 (환경변수에서 자동 로드)
            aws_secret_access_key: AWS 시크릿 키 (환경변수에서 자동 로드)
            region_name: AWS 리전
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")

        # AWS 클라이언트 초기화
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=region_name,
            )
            self.s3_client = session.client("s3")
            self.s3_resource = session.resource("s3")

            # 버킷 접근 테스트
            self._test_bucket_access()
            logger.info(f"S3 storage manager initialized for bucket: {bucket_name}")

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _test_bucket_access(self) -> None:
        """버킷 접근 권한 테스트"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            elif error_code == 403:
                raise ValueError(f"Access denied to bucket {self.bucket_name}")
            else:
                raise

    def _build_key(self, *path_parts: str) -> str:
        """S3 키 생성"""
        parts = [self.prefix] if self.prefix else []
        parts.extend(str(part).strip("/") for part in path_parts if part)
        return "/".join(parts)

    def _ensure_container_structure(self) -> None:
        """MLOps 컨테이너 구조 생성"""
        containers = [
            "raw-data/weather/kma",
            "raw-data/weather/external",
            "processed-data/features",
            "processed-data/labels",
            "models/experiments",
            "models/staging",
            "models/production",
            "artifacts/plots",
            "artifacts/reports",
            "artifacts/configs",
            "logs/training",
            "logs/inference",
            "logs/pipeline",
            "monitoring/metrics",
            "monitoring/alerts",
            "monitoring/evaluation",
            "monitoring/predictions",
        ]

        for container in containers:
            key = self._build_key(container, ".keep")
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            except ClientError:
                # 컨테이너가 없으면 생성
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=b"",
                    Metadata={"created_by": "mlops_pipeline", "created_at": datetime.utcnow().isoformat()},
                )
                logger.info(f"Created container: {container}")

    # Raw Data Methods
    def store_raw_weather_data(
        self,
        data: str,
        station_id: str,
        timestamp: datetime,
        source: str = "kma",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """원시 날씨 데이터 저장"""
        date_path = timestamp.strftime("%Y/%m/%d")
        time_suffix = timestamp.strftime("%H%M")

        key = self._build_key(
            "raw-data",
            "weather",
            source,
            station_id,
            date_path,
            f"{time_suffix}.txt"
        )

        # 메타데이터 추가
        s3_metadata = {
            "station_id": station_id,
            "timestamp": timestamp.isoformat(),
            "source": source,
            "data_size": str(len(data.encode("utf-8"))),
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data.encode("utf-8"),
            Metadata=s3_metadata,
            ContentType="text/plain",
        )

        logger.info(f"Stored raw weather data: {key}")
        return key

    def store_processed_features(
        self,
        df: pd.DataFrame,
        feature_set_name: str,
        version: str,
        timestamp: datetime
    ) -> str:
        """처리된 피처 데이터 저장"""
        date_path = timestamp.strftime("%Y/%m/%d")
        time_suffix = timestamp.strftime("%H%M")

        key = self._build_key(
            "processed-data",
            "features",
            feature_set_name,
            version,
            date_path,
            f"{time_suffix}.parquet"
        )

        # DataFrame을 parquet으로 저장
        buffer = df.to_parquet(index=False)

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer,
            Metadata={
                "feature_set": feature_set_name,
                "version": version,
                "timestamp": timestamp.isoformat(),
                "rows": str(len(df)),
                "columns": str(len(df.columns)),
                "column_names": ",".join(df.columns.tolist()),
            },
            ContentType="application/octet-stream",
        )

        logger.info(f"Stored processed features: {key}")
        return key

    # Model Methods
    def store_model_artifact(
        self,
        model_data: bytes,
        model_name: str,
        version: str,
        stage: str = "experiments",  # experiments, staging, production
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """모델 아티팩트 저장"""
        if stage not in ["experiments", "staging", "production"]:
            raise ValueError("Stage must be one of: experiments, staging, production")

        timestamp = datetime.utcnow()
        key = self._build_key(
            "models",
            stage,
            model_name,
            version,
            f"model_{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        )

        s3_metadata = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "created_at": timestamp.isoformat(),
            "size_bytes": str(len(model_data)),
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=model_data,
            Metadata=s3_metadata,
            ContentType="application/octet-stream",
        )

        logger.info(f"Stored model artifact: {key}")
        return key

    def write_text(self, key: str, content: str) -> None:
        """텍스트 파일을 S3에 저장"""
        full_key = self._build_key(key) if not key.startswith(self.prefix) else key

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=full_key,
            Body=content.encode("utf-8"),
            ContentType="text/plain"
        )
        logger.debug(f"Wrote text to S3: {full_key}")

    def read_text(self, key: str) -> str:
        """S3에서 텍스트 파일 읽기"""
        full_key = self._build_key(key) if not key.startswith(self.prefix) else key

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Key not found: {full_key}")
            raise

    def list_files_by_prefix(self, prefix: str) -> List[str]:
        """특정 접두사로 시작하는 파일들 목록 반환"""
        full_prefix = self._build_key(prefix) if not prefix.startswith(self.prefix) else prefix

        files = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix)

            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append(obj['Key'])

        except ClientError as e:
            logger.error(f"Failed to list files with prefix {full_prefix}: {e}")

        return files

    def write_json(self, key: str, data: Dict[str, Any]) -> None:
        """JSON 데이터를 S3에 저장"""
        content = json.dumps(data, indent=2, ensure_ascii=False)
        full_key = self._build_key(key) if not key.startswith(self.prefix) else key

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=full_key,
            Body=content.encode("utf-8"),
            ContentType="application/json"
        )
        logger.debug(f"Wrote JSON to S3: {full_key}")

    def read_json(self, key: str) -> Dict[str, Any]:
        """S3에서 JSON 파일 읽기"""
        content = self.read_text(key)
        return json.loads(content)

    def promote_model(self, model_key: str, target_stage: str) -> str:
        """모델을 다른 스테이지로 승격"""
        if target_stage not in ["staging", "production"]:
            raise ValueError("Target stage must be staging or production")

        # 원본 모델 메타데이터 가져오기
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=model_key)
        metadata = response["Metadata"]

        # 새로운 키 생성
        model_name = metadata.get("model_name", "unknown")
        version = metadata.get("version", "unknown")
        timestamp = datetime.utcnow()

        new_key = self._build_key(
            "models",
            target_stage,
            model_name,
            version,
            f"model_{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        )

        # 모델 복사
        copy_source = {"Bucket": self.bucket_name, "Key": model_key}
        metadata["stage"] = target_stage
        metadata["promoted_at"] = timestamp.isoformat()

        self.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket_name,
            Key=new_key,
            Metadata=metadata,
            MetadataDirective="REPLACE",
        )

        logger.info(f"Promoted model from {model_key} to {new_key}")
        return new_key

    # Monitoring and Logging Methods
    def store_training_log(
        self,
        log_data: Dict[str, Any],
        experiment_name: str,
        run_id: str
    ) -> str:
        """훈련 로그 저장"""
        timestamp = datetime.utcnow()
        key = self._build_key(
            "logs",
            "training",
            experiment_name,
            run_id,
            f"training_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )

        log_data["logged_at"] = timestamp.isoformat()

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(log_data, indent=2, default=str).encode("utf-8"),
            Metadata={
                "experiment_name": experiment_name,
                "run_id": run_id,
                "log_type": "training",
                "timestamp": timestamp.isoformat(),
            },
            ContentType="application/json",
        )

        logger.info(f"Stored training log: {key}")
        return key

    def store_inference_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        timestamp: datetime
    ) -> str:
        """추론 메트릭 저장"""
        date_path = timestamp.strftime("%Y/%m/%d")
        hour_suffix = timestamp.strftime("%H")

        key = self._build_key(
            "monitoring",
            "metrics",
            model_name,
            date_path,
            f"metrics_{hour_suffix}.json"
        )

        metrics_data = {
            "model_name": model_name,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
        }

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(metrics_data, indent=2).encode("utf-8"),
            Metadata={
                "model_name": model_name,
                "timestamp": timestamp.isoformat(),
                "metrics_count": str(len(metrics)),
            },
            ContentType="application/json",
        )

        logger.info(f"Stored inference metrics: {key}")
        return key

    def store_evaluation_results(
        self,
        evaluation_data: Dict[str, Any],
        model_name: str,
        model_version: str,
        timestamp: datetime
    ) -> str:
        """모델 평가 결과 저장"""
        date_path = timestamp.strftime("%Y/%m/%d")
        time_suffix = timestamp.strftime("%H%M%S")

        key = self._build_key(
            "monitoring",
            "evaluation",
            model_name,
            model_version,
            date_path,
            f"evaluation_{time_suffix}.json"
        )

        evaluation_data["stored_at"] = datetime.utcnow().isoformat()

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(evaluation_data, indent=2, default=str).encode("utf-8"),
            Metadata={
                "model_name": model_name,
                "model_version": model_version,
                "evaluation_timestamp": timestamp.isoformat(),
                "result_type": "model_evaluation",
            },
            ContentType="application/json",
        )

        logger.info(f"Stored evaluation results: {key}")
        return key

    def store_prediction_results(
        self,
        predictions: List[Dict[str, Any]],
        model_name: str,
        model_version: str,
        timestamp: datetime
    ) -> str:
        """예측 결과 저장"""
        date_path = timestamp.strftime("%Y/%m/%d")
        hour_suffix = timestamp.strftime("%H")

        key = self._build_key(
            "monitoring",
            "predictions",
            model_name,
            model_version,
            date_path,
            f"predictions_{hour_suffix}.json"
        )

        prediction_data = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": timestamp.isoformat(),
            "predictions": predictions,
            "count": len(predictions)
        }

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(prediction_data, indent=2, default=str).encode("utf-8"),
            Metadata={
                "model_name": model_name,
                "model_version": model_version,
                "prediction_timestamp": timestamp.isoformat(),
                "prediction_count": str(len(predictions)),
            },
            ContentType="application/json",
        )

        logger.info(f"Stored prediction results: {key}")
        return key

    # Query Methods
    def list_objects_by_prefix(self, prefix: str) -> List[Dict[str, Any]]:
        """특정 접두사로 객체 목록 조회"""
        full_prefix = self._build_key(prefix)

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix)

            objects = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                        })

            return objects
        except ClientError as e:
            logger.error(f"Error listing objects with prefix {prefix}: {e}")
            return []

    def get_latest_model(self, model_name: str, stage: str = "production") -> Optional[str]:
        """최신 모델 키 반환"""
        prefix = f"models/{stage}/{model_name}"
        objects = self.list_objects_by_prefix(prefix)

        if not objects:
            return None

        # 가장 최근에 수정된 객체 반환
        latest = max(objects, key=lambda x: x["last_modified"])
        return latest["key"]

    def download_object(self, key: str, local_path: Optional[str] = None) -> Union[bytes, str]:
        """S3 객체 다운로드"""
        try:
            if local_path:
                self.s3_client.download_file(self.bucket_name, key, local_path)
                logger.info(f"Downloaded {key} to {local_path}")
                return local_path
            else:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                return response["Body"].read()
        except ClientError as e:
            logger.error(f"Error downloading object {key}: {e}")
            raise

    def initialize_mlops_structure(self) -> None:
        """MLOps 디렉토리 구조 초기화"""
        logger.info("Initializing MLOps container structure...")
        self._ensure_container_structure()

        # README 파일 생성
        readme_content = """# MLOps S3 Container Structure

## Directory Layout

- **raw-data/**: 원시 데이터
  - weather/kma/: 기상청 API 데이터
  - weather/external/: 외부 날씨 데이터

- **processed-data/**: 처리된 데이터
  - features/: 피처 엔지니어링 결과
  - labels/: 레이블 데이터

- **models/**: 모델 아티팩트
  - experiments/: 실험 단계 모델
  - staging/: 스테이징 환경 모델
  - production/: 프로덕션 모델

- **artifacts/**: 부산물
  - plots/: 시각화 결과
  - reports/: 분석 리포트
  - configs/: 설정 파일

- **logs/**: 로그 데이터
  - training/: 훈련 로그
  - inference/: 추론 로그
  - pipeline/: 파이프라인 로그

- **monitoring/**: 모니터링 데이터
  - metrics/: 성능 메트릭
  - alerts/: 알림 로그
"""

        readme_key = self._build_key("README.md")
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=readme_key,
            Body=readme_content.encode("utf-8"),
            ContentType="text/markdown",
        )

        logger.info("MLOps container structure initialized successfully")