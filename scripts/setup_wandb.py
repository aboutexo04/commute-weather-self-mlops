#!/usr/bin/env python3
"""WandB 설정 도우미 스크립트."""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv가 설치되지 않았습니다. 'pip install python-dotenv'를 실행하세요.")
    sys.exit(1)


def check_wandb_setup():
    """WandB 설정 상태 확인"""
    print("🔍 WandB 설정 상태 확인 중...")

    # API 키 확인
    api_key = os.getenv("WANDB_API_KEY")
    if api_key and api_key != "your_wandb_api_key_here":
        print("✅ WANDB_API_KEY 설정됨")
        return True
    else:
        print("❌ WANDB_API_KEY가 설정되지 않았습니다.")
        return False


def setup_wandb_interactive():
    """대화형 WandB 설정"""
    print("\n🎯 WandB 설정을 시작합니다...")

    # WandB 로그인 시도
    try:
        import wandb
        print("📦 WandB 라이브러리가 설치되어 있습니다.")

        # 로그인 상태 확인
        if wandb.api.api_key:
            print("✅ 이미 WandB에 로그인되어 있습니다.")
            return True
        else:
            print("🔑 WandB 로그인이 필요합니다.")
            print("다음 중 하나를 선택하세요:")
            print("1. 브라우저에서 자동 로그인")
            print("2. API 키 직접 입력")

            choice = input("선택 (1 또는 2): ").strip()

            if choice == "1":
                try:
                    wandb.login()
                    print("✅ WandB 로그인 성공!")
                    return True
                except Exception as e:
                    print(f"❌ 자동 로그인 실패: {e}")
                    return False

            elif choice == "2":
                api_key = input("WandB API 키를 입력하세요: ").strip()
                if api_key:
                    # .env 파일 업데이트
                    update_env_file("WANDB_API_KEY", api_key)
                    print("✅ API 키가 .env 파일에 저장되었습니다.")
                    return True
                else:
                    print("❌ API 키가 입력되지 않았습니다.")
                    return False
            else:
                print("❌ 잘못된 선택입니다.")
                return False

    except ImportError:
        print("❌ WandB 라이브러리가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install wandb")
        return False


def update_env_file(key, value):
    """환경변수 파일 업데이트"""
    env_file = project_root / ".env"

    # .env 파일 읽기
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()

    # 기존 키 찾기 및 업데이트
    key_found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f'{key}="{value}"\n'
            key_found = True
            break

    # 키가 없으면 추가
    if not key_found:
        lines.append(f'{key}="{value}"\n')

    # 파일 쓰기
    with open(env_file, 'w') as f:
        f.writelines(lines)


def test_wandb_integration():
    """WandB 통합 테스트"""
    print("\n🧪 WandB 통합 테스트 중...")

    try:
        from commute_weather.tracking import WandBManager

        # 테스트 WandB 매니저 생성
        wandb_manager = WandBManager(
            project_name="commute-weather-test",
            tags=["setup-test"]
        )

        # 간단한 테스트 실행
        run = wandb_manager.start_run(
            run_name="setup_test",
            job_type="test",
            notes="WandB 설정 테스트"
        )

        # 테스트 메트릭 로깅
        wandb_manager.log_metrics({
            "test_metric": 42.0,
            "setup_success": 1.0
        })

        # 실행 종료
        wandb_manager.finish_run()

        print("✅ WandB 통합 테스트 성공!")
        print(f"🔗 실행 URL: {run.url if run else 'N/A'}")
        return True

    except Exception as e:
        print(f"❌ WandB 통합 테스트 실패: {e}")
        return False


def main():
    """메인 함수"""
    print("🚀 WandB 설정 도우미")
    print("=" * 50)

    # 1. 현재 설정 상태 확인
    if check_wandb_setup():
        print("✅ WandB가 이미 설정되어 있습니다.")

        # 테스트 실행 여부 확인
        test_choice = input("\n🧪 통합 테스트를 실행하시겠습니까? (y/n): ").strip().lower()
        if test_choice == 'y':
            if test_wandb_integration():
                print("\n🎉 모든 설정이 완료되었습니다!")
            else:
                print("\n⚠️  테스트에 실패했습니다. 설정을 확인하세요.")
        return

    # 2. 대화형 설정
    if setup_wandb_interactive():
        print("\n✅ WandB 설정이 완료되었습니다!")

        # 3. 통합 테스트
        if test_wandb_integration():
            print("\n🎉 모든 설정이 성공적으로 완료되었습니다!")
            print("\n🚀 이제 다음 명령어로 실시간 파이프라인을 실행할 수 있습니다:")
            print("   python scripts/run_realtime_mlops.py realtime")
        else:
            print("\n⚠️  통합 테스트에 실패했습니다.")
            print("문제가 지속되면 수동으로 설정을 확인하세요.")
    else:
        print("\n❌ WandB 설정에 실패했습니다.")
        print("문서를 참고하여 수동으로 설정하세요: docs/wandb_integration.md")


if __name__ == "__main__":
    main()