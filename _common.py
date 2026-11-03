"""조사·점검 스크립트 공용 헬퍼: API 키를 환경변수 또는 .streamlit/secrets.toml 에서 읽는다."""
import os
import sys
import tomllib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_key():
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
    try:
        with open(path, "rb") as f:
            key = tomllib.load(f).get("NVIDIA_API_KEY", "")
    except FileNotFoundError:
        key = ""
    if not key:
        sys.exit("NVIDIA_API_KEY 환경변수 또는 .streamlit/secrets.toml 이 필요합니다.")
    return key
