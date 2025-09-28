# Commute Weather Data Pipeline Specification

## 1. Scope and Objectives
- Centralize acquisition of commute-related weather observations from KMA typ01 API (primary) and optional third-party HTTP providers.
- Preserve immutable raw payloads for replay and auditing.
- Produce standardized, schema-validated datasets ready for feature engineering and modeling experiments.
- Enable reproducible, automated workflows that capture data lineage across ingestion, validation, transformation, and model training.

## 2. Data Sources
### 2.1 KMA typ01 Surface Observation API
- **Endpoint**: `https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php`
- **Authentication**: `authKey` query parameter; provided via `KMA_AUTH_KEY` environment variable.
- **Parameters**:
  - `tm1`, `tm2`: 12-digit timestamps (YYYYMMDDHHMM) bounding the request window.
  - `stn`: Station identifier (`KMA_STATION_ID`, default `108` for Seoul).
  - `help`: Optional metadata flag (default `0`).
- **Delivery Format**: Plain-text table with comment headers, varying column order.
- **Polling Cadence**: Hourly minimum; scheduler triggers at 07:00, 14:00–18:00.

### 2.2 Supplemental HTTP Weather API (Optional)
- **Endpoint**: `<base_url>/observations`
- **Authentication**: Bearer token header when `api_key` provided.
- **Parameters**:
  - `location`: Place identifier.
  - `hours`: Lookback window.
- **Delivery Format**: JSON array under `observations` key.
- **Usage**: Backup data source and ensemble experimentation.

### 2.3 Local Sample Payloads
- **Location**: `data/raw/` checked-in sample files for development and testing.
- **Format**: JSON or raw KMA text snapshots.
- **Usage**: Offline validation, regression tests.

## 3. Storage Layout
```
data/
├── raw/
│   ├── kma/<station_id>/<YYYY>/<MM>/<DD>/<HHMM>.txt
│   └── http/<location>/<YYYY>/<MM>/<DD>/<HH>/*.json
├── interim/
│   ├── weather_normalized/<source>/<YYYY>/<MM>/<DD>/<HH>.parquet
│   └── quality_reports/<source>/<YYYY>/<MM>/<DD>/<HH>.json
└── processed/
    ├── features/commute/<version>/<YYYYMMDD>.parquet
    └── labels/commute_baseline/<YYYYMMDD>.parquet
```
- **Partition Keys**: Station ID or location, UTC timestamp truncated to hour, pipeline version (`<version>` semantic).
- **File Formats**: Text for raw, columnar Parquet for normalized/feature sets, JSON for QA reports.
- **Versioning**: Introduce `data/_meta/version.json` to track schema revisions (see Section 6).
- **Remote Storage**: When `COMMUTE_S3_BUCKET` 환경변수가 설정되면 동일한 디렉터리 트리를 `s3://<bucket>/<prefix>/...` 형태로 생성합니다. (예: `s3://my-mlops-symun/raw/kma/...`)

## 4. Data Contracts
### 4.1 Raw Layer (Immutable)
- Preserve payload exactly as received.
- Attach metadata sidecar (`.meta.json`) containing request parameters, response headers, fetch timestamp, checksum.

### 4.2 Normalized Weather Observation Schema (`weather_normalized`)
Field | Type | Description | Source Mapping
----- | ---- | ----------- | --------------
`timestamp` | datetime64[ns, Asia/Seoul] | Observation time | KMA `tm` or HTTP `timestamp`
`station_id` | string | Weather station identifier | `stn` or provider field
`temperature_c` | float32 | Near-surface air temperature °C | `ta`/`temp`
`wind_speed_ms` | float32 | Wind speed m/s | `ws`/`wind_speed`
`precipitation_mm` | float32 | Hourly precip mm | `rn`/`precipitation_mm`
`relative_humidity` | float32 | % relative humidity | `hm`/`relative_humidity`
`precipitation_type` | category {`rain`,`snow`,`none`,`unknown`} | Derived from precipitation + temperature | heuristics
`ingested_at` | datetime64[ns, UTC] | Pipeline ingestion time | pipeline stamp
`source` | category {`kma`,`http`} | Origin | pipeline stamp

### 4.3 Feature Set Schema (`features/commute`)
Field | Type | Description
----- | ---- | -----------
`event_time` | datetime64[ns, Asia/Seoul] | Prediction timestamp (rounded hour)
`target_period` | category {`morning_commute`,`evening_commute`}
`temp_median_3h` | float32 | Median temperature over last 3 hours
`temp_range_3h` | float32 | Max-min temperature last 3 hours
`wind_peak_3h` | float32 | Max wind speed last 3 hours
`precip_total_3h` | float32 | Sum precipitation last 3 hours
`precip_type_dominant` | category {`rain`,`snow`,`none`}
`humidity_avg_3h` | float32 | Mean humidity last 3 hours
`baseline_comfort_score` | float32 | Current heuristic score for traceability
`label_bucket` | category {`excellent`,`good`,`uncomfortable`,`harsh`} | Derived from baseline or user feedback (future)
`data_version` | string | Schema version (`vYYYY.MM.DD`)

### 4.4 Label Dataset (`labels/commute_baseline`)
- Contains the heuristic comfort scores (continuous and categorical) per timestamp to serve as baseline targets until user feedback is available.

## 5. Pipeline Stages
1. **Ingest** (`pipelines.ingest.kma`)
   - Fetch KMA payload for configured station/time window.
   - Write raw text + metadata to `data/raw/kma/...`.
   - Emit ingestion event (Prefect/Dagster).
2. **Normalize** (`pipelines.normalize.weather`)
   - Parse raw payload → `WeatherObservation` objects.
   - Persist normalized dataframe to `data/interim/weather_normalized/...`.
   - Perform schema enforcement via Pandera/Great Expectations checkpoint.
3. **Data Quality**
   - Compute record counts, null ratios, sentinel values.
   - Save validation report JSON under `data/interim/quality_reports/...`.
   - Publish alert when critical checks fail.
4. **Feature Engineering** (`pipelines.features.commute`)
   - Aggregate normalized observations into rolling 3h windows for morning/evening prediction windows.
   - Join with baseline comfort score for consistency.
   - Write feature set to `data/processed/features/commute/...`.
5. **Label Generation**
   - Persist baseline comfort outputs as `labels` dataset for supervised training.
6. **Model Training** (`pipelines.train.baseline_ml`)
   - Consume features + labels, split train/validation.
   - Log metrics/artifacts to MLflow (Section 7).
7. **Model Registry & Promotion**
   - Register candidate models once they outperform baseline.
   - Attach feature/label versions and validation reports.

## 6. Schema Versioning & Metadata
- Maintain `schemas/weather_normalized_v1.yaml`, `schemas/features_commute_v1.yaml` with explicit field types and tolerances.
- Increment version suffix when breaking changes occur; update `data/_meta/version.json` with active versions.
- Data processing code reads version metadata to route to correct parser/validator.

## 7. Observability & Monitoring
- **Lineage**: Store Prefect/Dagster run IDs alongside artifacts (metadata JSON) to trace transformations.
- **Metrics**: Capture ingestion latency, validation pass rate, record counts, and drift statistics (via Evidently/Great Expectations).
- **Alerts**: Integration with Slack/Email for failed validation or missing data windows.

## 8. Security & Compliance
- Secrets managed through `.env` locally; plan migration to Secret Manager (AWS Secrets Manager/GCP Secret Manager) in production.
- Raw data stored with least-privilege access; encrypt at rest when external storage is used.
- Audit logs retained for 90 days.

## 9. Next Actions
1. Implement Prefect flow that materializes the ingest → normalize → feature path to match this spec.
2. Add Pandera schema definitions and integrate into CI for regression guardrails.
3. Establish DVC or LakeFS repository tracking for `data/` partitions referenced above.
