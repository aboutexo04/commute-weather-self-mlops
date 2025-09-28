"""Setup script for commute-weather MLOps system."""

from setuptools import setup, find_packages

setup(
    name="commute-weather",
    version="0.1.0",
    description="Baseline commute comfort scoring project",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31",
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pandas>=2.1",
        "pyarrow>=14.0",
        "prefect>=2.14",
        "scikit-learn>=1.3",
        "mlflow>=2.9",
        "boto3>=1.28",
        "s3fs>=2023.9",
        "wandb>=0.16",
        "schedule>=1.2",
        "python-dotenv>=1.0",
        "psutil>=5.9",
        "pytz>=2023.3",
        "numpy>=1.24",
        "joblib>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
        ]
    },
)