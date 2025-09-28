"""Storage helpers supporting both local disk and S3 backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .config import ProjectPaths


@dataclass(frozen=True)
class StorageManager:
    """Abstracts writing/listing artifacts across local FS and S3."""

    paths: ProjectPaths
    bucket: Optional[str] = None
    prefix: str = ""

    def _is_remote(self) -> bool:
        return bool(self.bucket)

    def _normalize_parts(self, parts: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        for part in parts:
            if not part:
                continue
            normalized.append(part.strip("/"))
        return normalized

    def _build_path(self, layer: str, *parts: str) -> str:
        normalized_parts = self._normalize_parts(parts)
        if self._is_remote():
            key_parts = self._normalize_parts([self.prefix]) + [layer] + normalized_parts
            key = "/".join(key_parts)
            return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}"

        base_map = {
            "raw": self.paths.data_raw,
            "interim": self.paths.data_interim,
            "processed": self.paths.data_processed,
            "models": self.paths.root / "models",
        }
        base = base_map[layer]
        return str(base.joinpath(*normalized_parts))

    # Public API -----------------------------------------------------------------

    def raw_path(self, *parts: str) -> str:
        return self._build_path("raw", *parts)

    def interim_path(self, *parts: str) -> str:
        return self._build_path("interim", *parts)

    def processed_path(self, *parts: str) -> str:
        return self._build_path("processed", *parts)

    def models_path(self, *parts: str) -> str:
        return self._build_path("models", *parts)

    # Write helpers --------------------------------------------------------------

    def write_text(self, path: str, content: str) -> None:
        if path.startswith("s3://"):
            import fsspec

            with fsspec.open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(content, encoding="utf-8")

    def write_json(self, path: str, payload: dict) -> None:
        self.write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def write_dataframe(self, path: str, dataframe: pd.DataFrame) -> str:
        if path.startswith("s3://"):
            dataframe.to_parquet(path, index=False)
        else:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(path_obj, index=False)
        return path

    def dump_joblib(self, path: str, obj: object) -> str:
        import joblib

        if path.startswith("s3://"):
            import fsspec

            with fsspec.open(path, "wb") as handle:
                joblib.dump(obj, handle)
        else:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, path_obj)
        return path

    # Listing helpers ------------------------------------------------------------

    def list_feature_files(self, version: str) -> List[str]:
        if self._is_remote():
            import s3fs

            fs = s3fs.S3FileSystem()
            base_parts = self._normalize_parts([self.prefix, "processed", "features", "commute", version])
            base_key = "/".join(base_parts)
            pattern = f"{self.bucket}/{base_key}/**/*.parquet"
            files = fs.glob(pattern)
            return [f"s3://{file}" for file in files]

        base_dir = Path(self.processed_path("features", "commute", version))
        if not base_dir.exists():
            return []
        return [str(path) for path in base_dir.rglob("*.parquet")]

    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def exists(self, path: str) -> bool:
        if path.startswith("s3://"):
            import s3fs

            fs = s3fs.S3FileSystem()
            return fs.exists(path.replace("s3://", ""))

        return Path(path).exists()

    def initialize_mlops_storage(self) -> None:
        """MLOps 스토리지 구조 초기화"""
        if self._is_remote():
            # S3 버킷에 MLOps 컨테이너 구조 생성
            from .storage.s3_manager import S3StorageManager
            s3_manager = S3StorageManager(bucket_name=self.bucket, prefix=self.prefix)
            s3_manager.initialize_mlops_structure()
        else:
            # 로컬 디렉토리 구조 생성
            local_dirs = [
                self.paths.data_raw / "weather" / "kma",
                self.paths.data_interim / "features",
                self.paths.data_processed / "features",
                self.paths.root / "models" / "experiments",
                self.paths.root / "models" / "staging",
                self.paths.root / "models" / "production",
                self.paths.root / "artifacts" / "plots",
                self.paths.root / "logs" / "training",
                self.paths.root / "monitoring" / "metrics",
            ]

            for dir_path in local_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)

            print("Local MLOps directory structure initialized")
