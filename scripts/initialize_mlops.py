#!/usr/bin/env python3
"""MLOps 시스템 초기화 스크립트."""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from commute_weather.config import ProjectPaths
from commute_weather.storage import StorageManager
from commute_weather.storage.s3_manager import S3StorageManager


def initialize_local_mlops():
    """로컬 MLOps 환경 초기화"""
    print("🚀 Initializing local MLOps environment...")

    paths = ProjectPaths.from_root(project_root)
    storage = StorageManager(paths)
    storage.initialize_mlops_storage()

    print("✅ Local MLOps environment initialized successfully!")
    print(f"📁 Project root: {paths.root}")
    print(f"📁 Data directory: {paths.data_raw}")
    print(f"📁 Models directory: {paths.root / 'models'}")


def initialize_s3_mlops(bucket_name: str, prefix: str = ""):
    """S3 MLOps 환경 초기화"""
    print(f"🚀 Initializing S3 MLOps environment in bucket: {bucket_name}")

    try:
        # S3 매니저로 컨테이너 구조 생성
        s3_manager = S3StorageManager(bucket_name=bucket_name, prefix=prefix)
        s3_manager.initialize_mlops_structure()

        # 테스트 데이터 업로드
        test_data = {
            "initialized_at": "2024-01-01T00:00:00Z",
            "version": "1.0.0",
            "environment": "development"
        }

        test_key = s3_manager._build_key("artifacts", "configs", "init_test.json")
        s3_manager.s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=str(test_data).encode("utf-8"),
            ContentType="application/json"
        )

        print("✅ S3 MLOps environment initialized successfully!")
        print(f"🪣 Bucket: {bucket_name}")
        print(f"📂 Prefix: {prefix or '(root)'}")
        print(f"🔗 Test file: s3://{bucket_name}/{test_key}")

    except Exception as e:
        print(f"❌ Failed to initialize S3 environment: {e}")
        sys.exit(1)


def test_s3_connection(bucket_name: str):
    """S3 연결 테스트"""
    print(f"🔍 Testing S3 connection to bucket: {bucket_name}")

    try:
        s3_manager = S3StorageManager(bucket_name=bucket_name)

        # 버킷 나열 테스트
        objects = s3_manager.list_objects_by_prefix("artifacts/configs")

        print("✅ S3 connection successful!")
        print(f"📊 Found {len(objects)} objects in artifacts/configs")

        if objects:
            print("📂 Recent objects:")
            for obj in objects[-3:]:  # 최근 3개만 표시
                print(f"   - {obj['key']} ({obj['size']} bytes)")

    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MLOps 환경 초기화")
    parser.add_argument("command", choices=["local", "s3", "test"], help="초기화 유형")
    parser.add_argument("--bucket", help="S3 버킷 이름")
    parser.add_argument("--prefix", default="", help="S3 키 접두사")

    args = parser.parse_args()

    if args.command == "local":
        initialize_local_mlops()

    elif args.command == "s3":
        if not args.bucket:
            # 환경변수에서 버킷 이름 확인
            bucket = os.getenv("COMMUTE_S3_BUCKET")
            if not bucket:
                print("❌ S3 bucket name required. Use --bucket or set COMMUTE_S3_BUCKET")
                sys.exit(1)
            args.bucket = bucket

        initialize_s3_mlops(args.bucket, args.prefix)

    elif args.command == "test":
        if not args.bucket:
            bucket = os.getenv("COMMUTE_S3_BUCKET")
            if not bucket:
                print("❌ S3 bucket name required. Use --bucket or set COMMUTE_S3_BUCKET")
                sys.exit(1)
            args.bucket = bucket

        test_s3_connection(args.bucket)


if __name__ == "__main__":
    main()