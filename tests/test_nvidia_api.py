"""nvidia_api.py 와 verify_app.py 테스트. 네트워크는 전부 mock, 키는 가짜 값."""
import base64
import io
import json
import os
import runpy
import urllib.error
from unittest import mock

import pytest

import nvidia_api as api

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "TEST_KEY"


class FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=b""):
    return urllib.error.HTTPError("https://example.invalid", code, "x", {}, io.BytesIO(body))


def test_redact_and_error_text_truncated():
    e = http_error(401, f"invalid {KEY}".encode() + b"a" * 2000)
    msg = api.http_error_text(e, KEY)
    assert KEY not in msg and "***" in msg
    assert len(msg) <= api.MAX_ERROR_CHARS + 20


def test_call_chat_uses_reasoning_content_when_content_empty():
    resp = FakeResp({"choices": [{"message": {"content": None, "reasoning_content": " 생각 "}}]})
    with mock.patch("urllib.request.urlopen", return_value=resp):
        assert api.call_chat("m", [], KEY) == "생각"


def test_fallback_skips_dead_models_and_redacts():
    seen = []

    def fake(req, timeout=None):
        model = json.loads(req.data)["model"]
        seen.append(model)
        if model == "dead":
            raise http_error(410)
        return FakeResp({"choices": [{"message": {"content": "ok"}}]})
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        assert api.call_chat_fallback(["dead", "alive"], [], KEY) == "ok"
    assert seen == ["dead", "alive"]

    def all_dead(req, timeout=None):
        raise RuntimeError(f"boom {KEY}")
    with mock.patch("urllib.request.urlopen", side_effect=all_dead):
        with pytest.raises(RuntimeError) as ei:
            api.call_chat_fallback(["a", "b"], [], KEY)
    assert KEY not in str(ei.value)


@pytest.mark.parametrize("model_key,has_cfg,max_steps", [
    ("flux.1-dev (고화질, 권장)", True, 50),
    ("flux.2-klein-4b (빠름)", False, 4),
])
def test_call_image_payload_per_model(model_key, has_cfg, max_steps):
    sent = {}

    def fake(req, timeout=None):
        sent.update(json.loads(req.data))
        return FakeResp({"artifacts": [{"base64": base64.b64encode(b"IMG").decode()}]})
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        assert api.call_image("p", 1024, 1024, 99, KEY, model_key) == b"IMG"
    assert ("cfg_scale" in sent) is has_cfg
    assert sent["steps"] == max_steps


def test_generate_poster_copy_parses_labels():
    raw = "제목: AI 교실\n기간: 매주 화\n내용: 주민센터 · 선착순"
    with mock.patch("urllib.request.urlopen",
                    return_value=FakeResp({"choices": [{"message": {"content": raw}}]})):
        assert api.generate_poster_copy("x", KEY) == ("AI 교실", "매주 화", "주민센터 · 선착순")


def test_verify_app_runs_with_mocked_api(monkeypatch, capsys):
    """verify_app.py 가 app.py 소스 exec 없이 nvidia_api 를 import 해서 끝까지 돈다."""
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.setattr("time.sleep", lambda s: None)

    def fake(req, timeout=None):
        if "genai" in req.full_url:
            return FakeResp({"artifacts": [{"base64": base64.b64encode(b"IMG").decode()}]})
        return FakeResp({"choices": [{"message": {"content": "제목: T\n기간: P\n내용: C"}}]})
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        runpy.run_path(os.path.join(ROOT, "verify_app.py"), run_name="__main__")
    assert "결과: 실패 0건" in capsys.readouterr().out
