#!/usr/bin/env python3
"""모델 성능 모니터링 서비스."""

import sys
import os
import logging
import time
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from commute_weather.ml.mlflow_integration import MLflowManager
from commute_weather.storage.s3_manager import S3StorageManager
from commute_weather.config import ProjectPaths

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/performance_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """모델 성능 모니터링 서비스"""

    def __init__(self):
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.s3_bucket = os.getenv("COMMUTE_S3_BUCKET")
        self.wandb_api_key = os.getenv("WANDB_API_KEY")

        # MLflow 매니저 설정
        self.mlflow_manager = MLflowManager(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name="scheduled-commute-mlops"
        )

        # S3 매니저 설정 (있는 경우)
        self.s3_manager = None
        if self.s3_bucket:
            project_root = Path(__file__).resolve().parents[1]
            paths = ProjectPaths.from_root(project_root)
            self.s3_manager = S3StorageManager(paths=paths, bucket=self.s3_bucket)

        logger.info("Performance monitor initialized")

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 수집"""

        logger.info("Collecting performance metrics")

        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "production_model": None,
                "recent_experiments": [],
                "performance_trends": {},
                "alerts": []
            }

            # 현재 프로덕션 모델 정보
            try:
                production_model = self.mlflow_manager.get_production_model("enhanced-commute-weather")
                if production_model:
                    metrics["production_model"] = {
                        "model_name": "enhanced-commute-weather",
                        "status": "available"
                    }
                else:
                    metrics["production_model"] = {
                        "model_name": "enhanced-commute-weather",
                        "status": "not_found"
                    }
                    metrics["alerts"].append("No production model found")

            except Exception as e:
                logger.warning(f"Failed to get production model: {e}")
                metrics["alerts"].append(f"Production model check failed: {str(e)}")

            # 최근 실험 성능
            try:
                best_run, model_uri = self.mlflow_manager.get_best_model_from_experiment()
                if best_run:
                    metrics["recent_experiments"].append({
                        "run_id": best_run.info.run_id,
                        "rmse": best_run.data.metrics.get("rmse"),
                        "mae": best_run.data.metrics.get("mae"),
                        "r2_score": best_run.data.metrics.get("r2_score"),
                        "start_time": best_run.info.start_time,
                        "status": best_run.info.status
                    })

            except Exception as e:
                logger.warning(f"Failed to get recent experiments: {e}")
                metrics["alerts"].append(f"Experiment metrics check failed: {str(e)}")

            # 성능 트렌드 분석
            try:
                trend_metrics = self._analyze_performance_trends()
                metrics["performance_trends"] = trend_metrics

            except Exception as e:
                logger.warning(f"Failed to analyze trends: {e}")
                metrics["alerts"].append(f"Trend analysis failed: {str(e)}")

            logger.info(f"Collected metrics with {len(metrics['alerts'])} alerts")
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "alerts": [f"Metrics collection failed: {str(e)}"]
            }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 분석"""

        try:
            # 최근 7일간의 실험 데이터 가져오기
            import mlflow

            # 최근 실험들 조회
            recent_runs = mlflow.search_runs(
                experiment_ids=[self.mlflow_manager.experiment.experiment_id],
                max_results=50,
                order_by=["start_time DESC"]
            )

            if recent_runs.empty:
                return {"status": "no_data", "message": "No recent runs found"}

            # 성능 메트릭 트렌드 계산
            rmse_values = recent_runs["metrics.rmse"].dropna()
            mae_values = recent_runs["metrics.mae"].dropna()
            r2_values = recent_runs["metrics.r2_score"].dropna()

            trends = {
                "total_runs": len(recent_runs),
                "runs_with_metrics": len(rmse_values),
                "rmse_stats": {
                    "mean": float(rmse_values.mean()) if not rmse_values.empty else None,
                    "std": float(rmse_values.std()) if not rmse_values.empty else None,
                    "min": float(rmse_values.min()) if not rmse_values.empty else None,
                    "max": float(rmse_values.max()) if not rmse_values.empty else None,
                },
                "mae_stats": {
                    "mean": float(mae_values.mean()) if not mae_values.empty else None,
                    "std": float(mae_values.std()) if not mae_values.empty else None,
                    "min": float(mae_values.min()) if not mae_values.empty else None,
                    "max": float(mae_values.max()) if not mae_values.empty else None,
                },
                "r2_stats": {
                    "mean": float(r2_values.mean()) if not r2_values.empty else None,
                    "std": float(r2_values.std()) if not r2_values.empty else None,
                    "min": float(r2_values.min()) if not r2_values.empty else None,
                    "max": float(r2_values.max()) if not r2_values.empty else None,
                }
            }

            # 성능 저하 감지
            if len(rmse_values) >= 5:
                recent_5_rmse = rmse_values.head(5).mean()
                overall_rmse = rmse_values.mean()

                if recent_5_rmse > overall_rmse * 1.1:  # 최근 성능이 10% 이상 저하
                    trends["performance_alert"] = {
                        "type": "degradation",
                        "message": f"Recent RMSE ({recent_5_rmse:.4f}) is {((recent_5_rmse/overall_rmse - 1) * 100):.1f}% worse than average",
                        "recent_avg": float(recent_5_rmse),
                        "overall_avg": float(overall_rmse)
                    }

            return trends

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def send_performance_report(self, metrics: Dict[str, Any]) -> None:
        """성능 리포트 전송"""

        try:
            # WandB에 메트릭 로깅
            if self.wandb_api_key:
                import wandb

                wandb.init(
                    project="commute-weather-monitoring",
                    name=f"performance_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=["monitoring", "performance"]
                )

                # 메트릭 로깅
                wandb_metrics = {
                    "alerts_count": len(metrics.get("alerts", [])),
                    "production_model_available": metrics.get("production_model", {}).get("status") == "available",
                    "recent_experiments_count": len(metrics.get("recent_experiments", []))
                }

                # 성능 통계 로깅
                trends = metrics.get("performance_trends", {})
                if "rmse_stats" in trends and trends["rmse_stats"].get("mean"):
                    wandb_metrics.update({
                        "rmse_mean": trends["rmse_stats"]["mean"],
                        "rmse_std": trends["rmse_stats"]["std"],
                        "mae_mean": trends["mae_stats"]["mean"],
                        "r2_mean": trends["r2_stats"]["mean"]
                    })

                # 알림 내용 로깅
                if metrics.get("alerts"):
                    wandb_metrics["alert_messages"] = "; ".join(metrics["alerts"])

                wandb.log(wandb_metrics)
                wandb.finish()

                logger.info("Performance report sent to WandB")

            # S3에 메트릭 저장
            if self.s3_manager:
                try:
                    self.s3_manager.store_monitoring_metrics(
                        metrics=metrics,
                        model_name="enhanced-commute-weather",
                        timestamp=datetime.now()
                    )
                    logger.info("Performance metrics stored to S3")

                except Exception as e:
                    logger.warning(f"Failed to store metrics to S3: {e}")

        except Exception as e:
            logger.error(f"Failed to send performance report: {e}")

    def check_model_health(self) -> Dict[str, Any]:
        """모델 건강성 체크"""

        logger.info("Checking model health")

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "checks": {},
            "recommendations": []
        }

        try:
            # 1. 프로덕션 모델 존재 확인
            production_model = self.mlflow_manager.get_production_model("enhanced-commute-weather")
            health_status["checks"]["production_model"] = production_model is not None

            if not production_model:
                health_status["overall_health"] = "warning"
                health_status["recommendations"].append("Deploy a production model")

            # 2. 최근 훈련 확인 (24시간 이내)
            best_run, _ = self.mlflow_manager.get_best_model_from_experiment()
            recent_training = False

            if best_run:
                last_training = datetime.fromtimestamp(best_run.info.start_time / 1000)
                hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
                recent_training = hours_since_training < 24

            health_status["checks"]["recent_training"] = recent_training

            if not recent_training:
                health_status["overall_health"] = "warning"
                health_status["recommendations"].append("Consider retraining - no recent training detected")

            # 3. 모델 성능 확인
            if best_run:
                rmse = best_run.data.metrics.get("rmse", float('inf'))
                performance_ok = rmse < 0.2  # RMSE 임계값

                health_status["checks"]["performance"] = performance_ok
                health_status["checks"]["rmse"] = rmse

                if not performance_ok:
                    health_status["overall_health"] = "critical"
                    health_status["recommendations"].append(f"Model performance degraded: RMSE {rmse:.4f}")

            # 4. 데이터 품질 확인 (S3에서)
            if self.s3_manager:
                try:
                    recent_quality = self._check_recent_data_quality()
                    health_status["checks"]["data_quality"] = recent_quality["status"] == "good"

                    if recent_quality["status"] != "good":
                        health_status["overall_health"] = "warning"
                        health_status["recommendations"].append("Data quality issues detected")

                except Exception as e:
                    logger.warning(f"Data quality check failed: {e}")
                    health_status["checks"]["data_quality"] = None

            # 전체 건강성 평가
            failed_checks = sum(1 for check in health_status["checks"].values()
                              if check is False)

            if failed_checks >= 2:
                health_status["overall_health"] = "critical"
            elif failed_checks >= 1:
                health_status["overall_health"] = "warning"

            logger.info(f"Model health check completed: {health_status['overall_health']}")
            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "error",
                "error": str(e),
                "checks": {},
                "recommendations": ["Fix monitoring system issues"]
            }

    def _check_recent_data_quality(self) -> Dict[str, Any]:
        """최근 데이터 품질 확인"""

        try:
            # 최근 24시간 품질 리포트 조회
            recent_files = self.s3_manager.list_recent_quality_reports(hours_back=24)

            if not recent_files:
                return {"status": "no_data", "message": "No recent quality reports"}

            # 최신 품질 점수들 수집
            quality_scores = []
            for file_path in recent_files[-10:]:  # 최근 10개
                try:
                    report = self.s3_manager.read_json(file_path)
                    if "feature_quality_score" in report:
                        quality_scores.append(report["feature_quality_score"])
                except:
                    continue

            if not quality_scores:
                return {"status": "no_data", "message": "No quality scores found"}

            avg_quality = sum(quality_scores) / len(quality_scores)

            if avg_quality >= 0.8:
                return {"status": "good", "average_quality": avg_quality}
            elif avg_quality >= 0.6:
                return {"status": "fair", "average_quality": avg_quality}
            else:
                return {"status": "poor", "average_quality": avg_quality}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_monitoring_cycle(self) -> None:
        """모니터링 사이클 실행"""

        logger.info("🔍 Running monitoring cycle")

        try:
            # 성능 메트릭 수집
            metrics = self.collect_performance_metrics()

            # 모델 건강성 체크
            health = self.check_model_health()

            # 통합 리포트 생성
            report = {
                "monitoring_cycle": datetime.now().isoformat(),
                "performance_metrics": metrics,
                "health_check": health,
                "summary": {
                    "overall_status": health["overall_health"],
                    "alerts_count": len(metrics.get("alerts", [])),
                    "recommendations_count": len(health.get("recommendations", []))
                }
            }

            # 리포트 전송
            self.send_performance_report(report)

            logger.info(f"Monitoring cycle completed - Status: {health['overall_health']}")

        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")

    def run_monitor(self) -> None:
        """모니터링 서비스 실행"""

        logger.info("🚀 Starting performance monitoring service")

        # 스케줄 설정
        schedule.every(30).minutes.do(self.run_monitoring_cycle)  # 30분마다 모니터링
        schedule.every().hour.at(":00").do(self.run_monitoring_cycle)  # 매시 정각

        logger.info("Performance monitoring scheduled:")
        logger.info("  - Every 30 minutes")
        logger.info("  - Every hour at :00")

        try:
            # 초기 모니터링 실행
            self.run_monitoring_cycle()

            # 스케줄 실행
            while True:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크

        except KeyboardInterrupt:
            logger.info("Performance monitor stopped by user")

        except Exception as e:
            logger.error(f"Performance monitor failed: {e}")
            raise


def main():
    """메인 실행 함수"""

    logger.info("🔍 Starting Model Performance Monitor")

    try:
        monitor = ModelPerformanceMonitor()
        monitor.run_monitor()

    except Exception as e:
        logger.error(f"Performance monitor failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()