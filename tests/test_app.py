"""app.py 동작 테스트. 실제 NVIDIA API는 호출하지 않고 urllib.request.urlopen 을 mock 한다.
비밀값은 모두 가짜(TEST_*)다."""
import io
import json
import os
import urllib.error
from unittest import mock

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
SERVER_KEY = "TEST_SERVER_KEY_nvapi-fake"
USER_KEY = "TEST_USER_KEY"
PASSWORD = "TEST_PASSWORD"


class FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def chat_ok(text="안녕하세요"):
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req)
        return FakeResp({"choices": [{"message": {"content": text}}]})
    return _urlopen, calls


def make_app(server_key=None, password=None):
    at = AppTest.from_file(APP, default_timeout=30)
    if server_key is not None:
        at.secrets["NVIDIA_API_KEY"] = server_key
    if password is not None:
        at.secrets["APP_PASSWORD"] = password
    return at


def sidebar_inputs(at):
    return {t.label: t for t in at.sidebar.text_input}


def all_text(at):
    parts = []
    for kind in ("error", "warning", "info", "caption", "markdown"):
        parts += [e.value for e in getattr(at, kind)]
        parts += [e.value for e in getattr(at.sidebar, kind)]
    return "\n".join(str(p) for p in parts)


def send_chat(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def auth_header(req):
    return req.get_header("Authorization")


def test_boots_without_secrets():
    at = make_app().run()
    assert not at.exception
    # 서버 키가 없으면 접근 암호 입력칸이 나오지 않는다
    assert "접근 암호 (서버 키 사용 시)" not in sidebar_inputs(at)


def test_server_key_never_in_widget_values():
    at = make_app(SERVER_KEY, PASSWORD).run()
    assert not at.exception
    for t in at.sidebar.text_input:
        assert SERVER_KEY not in (t.value or "")
    assert SERVER_KEY not in all_text(at)


def test_correct_password_uses_server_key():
    at = make_app(SERVER_KEY, PASSWORD).run()
    sidebar_inputs(at)["접근 암호 (서버 키 사용 시)"].set_value(PASSWORD).run()
    fake, calls = chat_ok("서버 키 응답")
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "안녕")
    assert not at.exception
    assert len(calls) == 1
    assert auth_header(calls[0]) == f"Bearer {SERVER_KEY}"
    assert [m["role"] for m in at.session_state.messages] == ["user", "assistant"]


def test_wrong_password_rejected():
    at = make_app(SERVER_KEY, PASSWORD).run()
    sidebar_inputs(at)["접근 암호 (서버 키 사용 시)"].set_value("WRONG").run()
    assert any("접근 암호가 맞지 않습니다" in e.value for e in at.sidebar.error)
    fake, calls = chat_ok()
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "안녕")
    assert calls == []
    assert any("API Key" in e.value for e in at.error)


def test_password_not_configured_server_key_unused_and_notice_shown():
    """APP_PASSWORD 미설정이면 서버 키를 쓰지 않는다(fail-closed) + 안내 문구가 보여야 한다."""
    at = make_app(SERVER_KEY, None).run()
    assert not at.exception
    assert "접근 암호 (서버 키 사용 시)" not in sidebar_inputs(at)
    assert "APP_PASSWORD" in all_text(at)
    fake, calls = chat_ok()
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "안녕")
    assert calls == []


def test_user_typed_key_takes_priority():
    at = make_app(SERVER_KEY, PASSWORD).run()
    sidebar_inputs(at)["NVIDIA API Key"].set_value(f"  {USER_KEY}  ").run()
    fake, calls = chat_ok()
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "안녕")
    assert len(calls) == 1
    assert auth_header(calls[0]) == f"Bearer {USER_KEY}"


def test_chat_failure_keeps_message_order_and_redacts_key():
    at = make_app().run()
    sidebar_inputs(at)["NVIDIA API Key"].set_value(USER_KEY).run()

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {},
                                     io.BytesIO(f"bad key {USER_KEY} ".encode() + b"x" * 1000))
    with mock.patch("urllib.request.urlopen", side_effect=boom):
        send_chat(at, "첫 질문")
    assert at.session_state.messages == []
    err = " ".join(e.value for e in at.error)
    assert "API 오류 (500)" in err and USER_KEY not in err
    assert len(err) < 400

    def timeout(req, timeout=None):
        raise TimeoutError("timed out")
    with mock.patch("urllib.request.urlopen", side_effect=timeout):
        send_chat(at, "두 번째 질문")
    assert at.session_state.messages == []

    fake, calls = chat_ok("정상 응답")
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "세 번째 질문")
    roles = [m["role"] for m in at.session_state.messages]
    assert roles == ["user", "assistant"]
    sent = json.loads(calls[0].data)["messages"]
    assert [m["content"] for m in sent] == ["세 번째 질문"]


def test_chat_input_length_limit():
    at = make_app().run()
    sidebar_inputs(at)["NVIDIA API Key"].set_value(USER_KEY).run()
    fake, calls = chat_ok()
    with mock.patch("urllib.request.urlopen", side_effect=fake):
        send_chat(at, "가" * 8001)
    assert calls == []
    assert any("너무 깁니다" in e.value for e in at.error)
    assert at.session_state.messages == []


# ---------- 업로드 크기 상한 (app.py 함수 단위) ----------

@pytest.fixture(scope="module")
def app_funcs():
    """app.py 를 한 번 실행해 전역 함수(safe_open_image 등)를 꺼낸다."""
    at = make_app().run()
    assert not at.exception
    import nvidia_api  # noqa: F401  (분리된 모듈이 있으면 import 가능해야 함)
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_under_test", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # bare 모드 streamlit — 경고만 출력
    return mod


def _png_bytes(size):
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format="PNG")
    return buf.getvalue()


def test_upload_size_limit(app_funcs):
    small = _png_bytes((64, 64))
    assert app_funcs.safe_open_image(small).size == (64, 64)
    with pytest.raises(ValueError, match="너무 큽니다"):
        app_funcs.safe_open_image(b"\0" * (app_funcs.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(ValueError, match="읽을 수 없습니다"):
        app_funcs.safe_open_image(b"not an image")


def test_upload_pixel_limit(app_funcs):
    # 파일은 작지만 픽셀 수가 상한의 2배를 넘는 이미지(디컴프레션 봄브) → 거부
    for size in ((8000, 7000), (6000, 6000)):  # 상한 2배 초과 / 1~2배 구간(PIL은 경고만 냄)
        big = _png_bytes(size)
        assert len(big) < app_funcs.MAX_UPLOAD_BYTES
        with pytest.raises(ValueError):
            app_funcs.safe_open_image(big)


def test_streamlit_upload_cap_configured():
    import tomllib
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["server"]["maxUploadSize"] <= 10


def test_compose_poster_with_logo_and_qr(app_funcs):
    base = _png_bytes((512, 768))
    out = app_funcs.compose_poster(
        base, "AI 교실", "매주 화요일", "주민센터 · 선착순 30명", "중앙", "하단", 0,
        app_funcs.POSTER_FONTS["나눔고딕 (기본, 산세리프)"], (255, 255, 255, 255), (240, 240, 240, 255),
        "약하게", "얇게", (0, 0, 0, 255), "반투명",
        logo_bytes=_png_bytes((100, 100)), qr_image=app_funcs.generate_qr_image("https://example.com"),
    )
    assert Image.open(io.BytesIO(out)).size == (512, 768)
    with pytest.raises(ValueError):
        app_funcs.compose_poster(base, "t", "", "", "중앙", "하단", 0,
                                 app_funcs.POSTER_FONTS["나눔고딕 (기본, 산세리프)"],
                                 (255, 255, 255, 255), (255, 255, 255, 255), "없음", "없음",
                                 (0, 0, 0, 255), "사용 안 함", logo_bytes=b"broken")
