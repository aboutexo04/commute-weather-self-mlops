"""Storage managers for the commute weather MLOps system."""

from .s3_manager import S3StorageManager

# Alias for compatibility
StorageManager = S3StorageManager

__all__ = ["S3StorageManager", "StorageManager"]