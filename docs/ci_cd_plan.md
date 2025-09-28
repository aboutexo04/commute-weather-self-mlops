# CI/CD Automation Plan for Commute Weather MLOps

## 1. Objectives
- Guarantee that data contracts, unit tests, and Prefect flows remain healthy before merging.
- Automate model evaluation and artifact registration once feature data meets minimum volume thresholds.
- Deploy FastAPI + inference services with environment-specific gating.

## 2. Pipeline Stages
1. **Static Checks**
   - `ruff` or `flake8` for linting (to be added).
   - `mypy` for type checking of new ML pipeline modules.
2. **Unit + Integration Tests**
   - `pytest` covering heuristic baseline and new ML tasks (with fixtures leveraging sample data).
   - Prefect flow dry-run using `prefect deployment build --skip-upload` to verify graph integrity.
3. **Data Quality Gate**
   - Execute `python -m commute_weather.validate --source kma` (CLI to implement) which runs Pandera/Great-Expectations checks on latest normalized batch.
4. **Model Training + Evaluation**
   - Invoke `python -m commute_weather.pipelines.ml.run_flow --mode train` to run Prefect flow locally or via Prefect Cloud agent.
   - Compare metrics logged to MLflow against stored baseline threshold (e.g., RMSE <= 8.0).
5. **Artifact Packaging**
   - Build Docker image tagged with Git SHA, embedding latest approved model artifact.
   - Publish to container registry (GHCR/ECR) guarded by branch protection.
6. **Deployment**
   - Staging deploy triggered on `main` merges; production deploy requires manual approval.
   - Use Infrastructure as Code (Terraform/CDKTF) to manage environment parity.

## 3. GitHub Actions Workflow Skeleton
```yaml
name: mlops-ci

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Lint
        run: |
          ruff check .
          mypy src
      - name: Unit tests
        run: pytest
      - name: Prefect flow dry-run
        run: prefect deployment build src/commute_weather/pipelines/ml/flow.py:build_commute_training_flow --name commute-train --skip-upload
      - name: Data quality smoke test
        run: python -m commute_weather.tools.data_quality --source kma --limit 10

  train-and-log:
    needs: build-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Run Prefect training flow
        env:
          KMA_AUTH_KEY: ${{ secrets.KMA_AUTH_KEY }}
          PREFECT_API_KEY: ${{ secrets.PREFECT_API_KEY }}
          PREFECT_API_URL: ${{ secrets.PREFECT_API_URL }}
        run: |
          prefect deployment run 'commute-weather-ml-training/commute-train' --params '{"target_period": "morning_commute"}'
      - name: Export latest metrics
        run: python scripts/export_mlflow_metrics.py --output metrics.json
      - name: Upload metrics artifact
        uses: actions/upload-artifact@v3
        with:
          name: mlflow-metrics
          path: metrics.json
```

## 4. Secrets & Configuration
- Store `KMA_AUTH_KEY`, `PREFECT_API_KEY`, `MLFLOW_TRACKING_URI`, and container registry credentials as encrypted repository secrets.
- Introduce `.env.ci` template for local GitHub Actions testing via `act`.

## 5. Next Steps
- Implement CLI helpers referenced in jobs (`data_quality`, `export_mlflow_metrics`).
- Add Ruff/Mypy configuration to `pyproject.toml` for reproducible linting.
- Containerize FastAPI + Prefect agent for staging deployments.
- Wire GitHub Environments for staging/production with manual approval gates.
