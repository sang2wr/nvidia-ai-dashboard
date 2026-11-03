import hmac
import io
import os
import urllib.request
import urllib.error

import qrcode
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from nvidia_api import (  # API 호출부·모델 설정은 nvidia_api.py 로 분리
    IMAGE_MODELS, MAX_ERROR_CHARS, MODEL_CATEGORIES, REASONING_MODELS,
    call_chat, call_image, generate_english_prompt, generate_poster_copy,
    has_hangul, http_error_text, redact, translate_to_en,
)

st.set_page_config(page_title="NVIDIA AI 모델 플레이그라운드", page_icon="🟢", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZES = {
    # 쇼츠·아바타는 768×1344를 쓴다(9:16에 가깝고 flux.1-dev가 받는 최대 높이). 2026-09-06 추가
    "쇼츠·아바타 세로 (768×1344)": (768, 1344),
    "정사각형 (1024×1024)": (1024, 1024),
    "세로형 (832×1216)": (832, 1216),
    "가로형 (1216×832)": (1216, 832),
}

POSTER_FONTS = {
    "나눔고딕 (기본, 산세리프)": {
        "regular": os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf"),
        "bold": os.path.join(BASE_DIR, "fonts", "NanumGothicBold.ttf"),
    },
    "Noto Sans KR (산세리프)": {
        "regular": os.path.join(BASE_DIR, "fonts", "NotoSansKR-Regular.ttf"),
        "bold": os.path.join(BASE_DIR, "fonts", "NotoSansKR-Bold.ttf"),
    },
    "나눔명조 (세리프, 격식있는 느낌)": {
        "regular": os.path.join(BASE_DIR, "fonts", "NanumMyeongjo.ttf"),
        "bold": os.path.join(BASE_DIR, "fonts", "NanumMyeongjoBold.ttf"),
    },
}

SHADOW_LEVELS = {"없음": 0, "약하게": 2, "강하게": 5}
BORDER_LEVELS = {"없음": 0, "얇게": 2, "두껍게": 4}
BAR_LEVELS = {"사용 안 함": None, "반투명": 120, "진하게": 190}


# ---------- 보안·입력 제한 ----------

MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # 로고 업로드 상한
MAX_UPLOAD_PIXELS = 25_000_000          # 디컴프레션 봄브 방지
MAX_CHAT_INPUT_CHARS = 8000
MAX_CHAT_HISTORY = 40                   # API로 보내는 최근 메시지 수 상한
MAX_PROMPT_CHARS = 800
MAX_QR_CHARS = 1000

Image.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS


def resolve_api_key():
    """사용자가 직접 입력한 키가 우선. 서버 키(st.secrets)는 브라우저로 내려보내지 않고
    APP_PASSWORD 를 맞게 입력한 경우에만 서버 쪽에서 사용한다(미설정 시 서버 키는 쓰지 않음)."""
    typed = st.sidebar.text_input("NVIDIA API Key", type="password",
                                  help="본인 키를 입력하거나, 아래 접근 암호를 입력하면 서버 키를 사용합니다.")
    if typed.strip():
        return typed.strip()
    try:
        server_key = st.secrets.get("NVIDIA_API_KEY", "")
        app_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:  # secrets.toml 없음
        return ""
    if not server_key:
        return ""
    if not app_password:
        # 서버 키는 있지만 APP_PASSWORD 미설정 → 서버 키는 쓰지 않는다(fail-closed). 이유를 화면에 알린다.
        st.sidebar.warning("공용(서버) 키는 관리자가 접근 암호(APP_PASSWORD)를 설정해야 사용할 수 있습니다. "
                           "지금은 위에 본인 NVIDIA API Key를 입력해야 사용할 수 있습니다.")
        return ""
    given = st.sidebar.text_input("접근 암호 (서버 키 사용 시)", type="password")
    if given and hmac.compare_digest(given.encode("utf-8"), str(app_password).encode("utf-8")):
        return server_key
    if given:
        st.sidebar.error("접근 암호가 맞지 않습니다.")
    return ""


def safe_open_image(data):
    """업로드 이미지를 크기·픽셀 수를 제한해서 연다. 실패하면 ValueError."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("이미지 파일이 너무 큽니다 (최대 5MB).")
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"이미지를 읽을 수 없습니다: {type(e).__name__}") from e
    # PIL은 MAX_IMAGE_PIXELS의 1~2배 구간에선 경고만 내고 그대로 연다 → 디코딩 전에 직접 확인
    if img.width * img.height > MAX_UPLOAD_PIXELS:
        raise ValueError(f"이미지 해상도가 너무 큽니다 (최대 {MAX_UPLOAD_PIXELS // 1_000_000}MP).")
    try:
        img.load()
    except Exception as e:
        raise ValueError(f"이미지를 읽을 수 없습니다: {type(e).__name__}") from e
    return img


# ---------- 포스터 합성 ----------

def fit_font(draw, text, font_path, max_width, start_size, min_size=14):
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, min_size)


def wrap_text(draw, text, font, max_width):
    if not text:
        return []
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    final_lines = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            final_lines.append(line)
            continue
        buf = ""
        for ch in line:
            trial = buf + ch
            bbox = draw.textbbox((0, 0), trial, font=font)
            if (bbox[2] - bbox[0]) <= max_width or not buf:
                buf = trial
            else:
                final_lines.append(buf)
                buf = ch
        if buf:
            final_lines.append(buf)
    return final_lines


def compose_poster(
    image_bytes, title, period, content,
    align, position_preset, offset_pct,
    font_paths, title_color, body_color,
    shadow_level, border_level, border_color, bar_level,
    logo_bytes=None, logo_position="우상단", logo_size_pct=15,
    qr_image=None, qr_position="우하단", qr_size_pct=12,
):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    padding_x = int(w * 0.07)
    max_text_w = w - padding_x * 2
    gap = int(h * 0.015)

    items = []  # (text, font, role)
    if title:
        font = fit_font(draw, title, font_paths["bold"], max_text_w, int(h * 0.075))
        items.append((title, font, "title"))
    if period:
        font = ImageFont.truetype(font_paths["regular"], int(h * 0.032))
        items.append((period, font, "body"))
    if content:
        font = ImageFont.truetype(font_paths["regular"], int(h * 0.026))
        for line in wrap_text(draw, content, font, max_text_w):
            items.append((line, font, "body"))

    if not items:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    heights = []
    total_h = 0
    for text, font, _role in items:
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
        heights.append(th)
        total_h += th + gap
    total_h -= gap

    bar_pad = int(h * 0.035)
    bar_h = total_h + bar_pad * 2

    if position_preset == "상단":
        base_top = 0
    elif position_preset == "중앙":
        base_top = (h - bar_h) // 2
    else:
        base_top = h - bar_h

    offset = int(h * offset_pct / 100)
    bar_top = max(0, min(h - bar_h, base_top + offset))

    bar_alpha = BAR_LEVELS.get(bar_level)
    if bar_alpha:
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle([0, bar_top, w, bar_top + bar_h], fill=(0, 0, 0, bar_alpha))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

    shadow_offset = SHADOW_LEVELS.get(shadow_level, 0)
    border_width = BORDER_LEVELS.get(border_level, 0)

    y = bar_top + bar_pad
    for (text, font, role), th in zip(items, heights):
        color = title_color if role == "title" else body_color
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        if align == "좌측":
            x = padding_x
        elif align == "우측":
            x = w - padding_x - tw
        else:
            x = (w - tw) // 2

        if shadow_offset:
            draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 160))
        if border_width:
            draw.text((x, y), text, font=font, fill=color, stroke_width=border_width, stroke_fill=border_color)
        else:
            draw.text((x, y), text, font=font, fill=color)
        y += th + gap

    if logo_bytes:
        logo_img = safe_open_image(logo_bytes)
        img = paste_overlay(img, logo_img, logo_position, logo_size_pct)

    if qr_image is not None:
        img = paste_overlay(img, qr_image, qr_position, qr_size_pct)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


CORNER_POSITIONS = ["좌상단", "우상단", "좌하단", "우하단"]


def generate_qr_image(data):
    qr = qrcode.QRCode(border=1)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGBA")


def paste_overlay(img, overlay_img, position, size_pct, padding_pct=0.03):
    w, h = img.size
    overlay_img = overlay_img.convert("RGBA")
    max_dim = max(1, int(min(w, h) * size_pct / 100))
    ow, oh = overlay_img.size
    scale = max_dim / max(ow, oh)
    new_size = (max(1, int(ow * scale)), max(1, int(oh * scale)))
    resized = overlay_img.resize(new_size, Image.LANCZOS)

    pad = int(min(w, h) * padding_pct)
    if position == "좌상단":
        xy = (pad, pad)
    elif position == "우상단":
        xy = (w - new_size[0] - pad, pad)
    elif position == "좌하단":
        xy = (pad, h - new_size[1] - pad)
    else:
        xy = (w - new_size[0] - pad, h - new_size[1] - pad)

    img.paste(resized, xy, resized)
    return img


def upscale_image(image_bytes, factor):
    img = Image.open(io.BytesIO(image_bytes))
    new_size = (int(img.width * factor), int(img.height * factor))
    img = img.resize(new_size, Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


st.title("🟢 NVIDIA AI 모델 플레이그라운드")

with st.sidebar:
    st.header("설정")

    api_key = resolve_api_key()
    st.caption("🔒 입력한 내용(채팅·이미지 설명·포스터 아이디어)은 NVIDIA API로 전송됩니다. "
               "개인정보·민감정보는 입력하지 마세요.")

    mode = st.radio("모드", ["💬 텍스트 채팅", "🎨 이미지 생성", "🖼️ 포스터 생성"])

    st.divider()

    if mode == "💬 텍스트 채팅":
        category = st.selectbox("작업 유형", list(MODEL_CATEGORIES.keys()))
        model_options = MODEL_CATEGORIES[category]
        model_labels = [f"{name}  —  {desc}" for name, desc in model_options]
        selected_label = st.selectbox("모델 선택", model_labels)
        model = model_options[model_labels.index(selected_label)][0]

        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max tokens", 64, 4096, 768, 64)
        if model.startswith(REASONING_MODELS):
            st.caption("⚠️ 추론(reasoning) 모델이라 max_tokens가 낮으면 사고과정만 나오고 답이 잘릴 수 있습니다.")

        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    elif mode == "🎨 이미지 생성":
        st.caption("모델: flux.1-dev(고화질) / flux.2-klein-4b(빠름) 중 선택")
        size_label = st.selectbox("이미지 비율", list(IMAGE_SIZES.keys()))
        steps = st.slider("Steps (품질/속도)", 1, 50, 32, 1, help="flux.1-dev는 30~35 권장. klein은 4에서 자동으로 잘립니다.")
        st.caption("⏱️ 이미지 1장 생성에 수 초 정도 걸립니다.")

    else:
        st.caption("모델: FLUX.2-klein(이미지) + ising-calibration(문구/번역, 실패 시 자동 대체)")
        poster_size_label = st.selectbox("포스터 비율", list(IMAGE_SIZES.keys()), index=1)
        poster_steps = st.slider("Steps (품질/속도, 최대 4)", 1, 4, 4, 1)

        st.markdown("**문구 배치**")
        poster_align = st.selectbox("가로 정렬", ["좌측", "중앙", "우측"], index=1)
        poster_position = st.selectbox("세로 위치", ["상단", "중앙", "하단"], index=2)
        poster_offset = st.slider("위치 미세조정 (%)", -20, 20, 0, 1)

        st.markdown("**글꼴 · 색상**")
        poster_font_label = st.selectbox("폰트", list(POSTER_FONTS.keys()))
        poster_title_color = st.color_picker("제목 글자색", "#FFFFFF")
        poster_body_color = st.color_picker("기간/내용 글자색", "#F0F0F0")

        st.markdown("**효과**")
        poster_shadow_level = st.selectbox("그림자", list(SHADOW_LEVELS.keys()), index=1)
        poster_border_level = st.selectbox("테두리", list(BORDER_LEVELS.keys()), index=0)
        poster_border_color = st.color_picker("테두리색", "#000000") if poster_border_level != "없음" else "#000000"
        poster_bar_level = st.selectbox("텍스트 배경바", list(BAR_LEVELS.keys()), index=1)

        st.markdown("**로고 · QR코드**")
        poster_logo_file = st.file_uploader("로고 이미지 (선택, 배경투명 PNG 권장)", type=["png", "jpg", "jpeg"])
        if poster_logo_file:
            poster_logo_position = st.selectbox("로고 위치", CORNER_POSITIONS, index=1, key="logo_pos")
            poster_logo_size = st.slider("로고 크기 (%)", 5, 40, 15, 1, key="logo_size")
        else:
            poster_logo_position, poster_logo_size = "우상단", 15

        poster_qr_enabled = st.checkbox("QR 코드 추가")
        if poster_qr_enabled:
            poster_qr_data = st.text_input("QR 코드에 담을 URL 또는 텍스트", max_chars=MAX_QR_CHARS)
            poster_qr_position = st.selectbox("QR 위치", CORNER_POSITIONS, index=3, key="qr_pos")
            poster_qr_size = st.slider("QR 크기 (%)", 5, 30, 12, 1, key="qr_size")
        else:
            poster_qr_data, poster_qr_position, poster_qr_size = "", "우하단", 12

        poster_upscale = st.selectbox("업스케일", ["없음", "x2", "x4"], index=1)
        st.caption("⏱️ 전체 파이프라인(번역→이미지→문구→합성→업스케일)에 30초~1분 정도 걸립니다.")


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


if mode == "💬 텍스트 채팅":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
            st.stop()
        if len(user_input) > MAX_CHAT_INPUT_CHARS:
            st.error(f"메시지가 너무 깁니다 (최대 {MAX_CHAT_INPUT_CHARS}자).")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                try:
                    full_response = call_chat(
                        model, st.session_state.messages[-MAX_CHAT_HISTORY:], api_key,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                except urllib.error.HTTPError as e:
                    st.session_state.messages.pop()  # 실패한 질문이 남아 user 메시지가 연속되지 않게
                    st.error(http_error_text(e, api_key))
                    st.stop()
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"요청 실패: {redact(e, api_key)[:MAX_ERROR_CHARS]}")
                    st.stop()

            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif mode == "🎨 이미지 생성":
    if "images" not in st.session_state:
        st.session_state.images = []

    model_key = st.selectbox("이미지 모델", list(IMAGE_MODELS.keys()), index=0)
    prompt_in = st.text_area(
        "이미지 설명 (한글로 써도 됩니다)",
        placeholder="예: 노을 지는 바닷가 나무 데크길, 분홍 시폰 원피스를 입은 20대 여성, 전신",
        height=100,
        max_chars=2000,
        key="prompt_in",
    )

    # 한글이 들어오면 영어 프롬프트로 바꿔 바로 아래에 보여준다(수정 가능).
    # 같은 문장에 대해 두 번 호출하지 않도록 원문을 기억해 둔다.
    prompt = prompt_in
    if has_hangul(prompt_in) and api_key:
        if st.session_state.get("tr_src") != prompt_in:
            with st.spinner("영어 프롬프트로 변환 중..."):
                try:
                    st.session_state.tr_out = translate_to_en(prompt_in, api_key)
                    st.session_state.tr_src = prompt_in
                except Exception as e:
                    st.warning(f"번역 실패({redact(e, api_key)[:MAX_ERROR_CHARS]}) — 입력한 한글 그대로 씁니다.")
                    st.session_state.tr_out = prompt_in
                    st.session_state.tr_src = prompt_in
        prompt = st.text_area(
            "영어 프롬프트 (자동 변환됨 · 직접 고쳐도 됩니다)",
            value=st.session_state.get("tr_out", ""),
            height=100,
            key="prompt_en",
        )
    st.caption("프롬프트는 800자에서 잘립니다 — 중요한 내용(의상·구도)을 앞에 쓰세요.")
    generate = st.button("🎨 이미지 생성", type="primary")

    if generate:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
            st.stop()
        if not prompt.strip():
            st.error("이미지 설명을 입력해주세요.")
            st.stop()
        prompt = prompt[:MAX_PROMPT_CHARS]

        width, height = IMAGE_SIZES[size_label]
        with st.spinner("이미지 생성 중... (수 초 소요)"):
            try:
                img_bytes = call_image(prompt, width, height, steps, api_key, model_key)
                st.session_state.images.insert(0, {"prompt": prompt, "bytes": img_bytes})
            except urllib.error.HTTPError as e:
                st.error(http_error_text(e, api_key))
            except Exception as e:
                st.error(f"요청 실패: {redact(e, api_key)[:MAX_ERROR_CHARS]}")

    for i, item in enumerate(st.session_state.images):
        st.image(item["bytes"], caption=item["prompt"], width=512)
        st.download_button(
            "다운로드",
            data=item["bytes"],
            file_name=f"nvidia_flux_{i}.jpg",
            mime="image/jpeg",
            key=f"dl_{i}",
        )
        st.divider()

else:
    for key, default in [
        ("poster_en_prompt", ""),
        ("poster_raw_image", None),
        ("poster_title", ""),
        ("poster_period", ""),
        ("poster_content", ""),
        ("poster_composed", None),
        ("poster_final", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    idea = st.text_area(
        "① 포스터 아이디어를 입력하세요 (한글로, 자유롭게 — 행사/교육명, 일시, 장소, 대상 등을 포함하면 좋습니다)",
        placeholder="예: 중장년 대상 AI 교육 안내 포스터, 8월 매주 화요일 오후 2시, 주민센터 강당, 따뜻한 파스텔톤 분위기",
        height=100,
        max_chars=5000,
    )

    col1, col2 = st.columns(2)
    with col1:
        step1 = st.button("1단계: 영어 프롬프트 생성", use_container_width=True)
    with col2:
        step3 = st.button("3단계: 포스터 문구 생성", use_container_width=True)

    if step1:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
        elif not idea.strip():
            st.error("아이디어를 입력해주세요.")
        else:
            with st.spinner("영어 프롬프트 생성 중..."):
                try:
                    st.session_state.poster_en_prompt = generate_english_prompt(idea, api_key)
                except Exception as e:
                    st.error(f"요청 실패: {redact(e, api_key)[:MAX_ERROR_CHARS]}")

    st.session_state.poster_en_prompt = st.text_area(
        "② 영어 프롬프트 (자동 생성됨, 직접 수정 가능)",
        value=st.session_state.poster_en_prompt,
        height=80,
    )

    step2 = st.button("2단계: 이미지 생성", type="primary")
    if step2:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
        elif not st.session_state.poster_en_prompt.strip():
            st.error("영어 프롬프트를 먼저 생성하거나 입력해주세요.")
        else:
            width, height = IMAGE_SIZES[poster_size_label]
            with st.spinner("이미지 생성 중... (수 초 소요)"):
                try:
                    st.session_state.poster_raw_image = call_image(
                        st.session_state.poster_en_prompt, width, height, poster_steps, api_key
                    )
                    st.session_state.poster_composed = None
                    st.session_state.poster_final = None
                except Exception as e:
                    st.error(f"요청 실패: {redact(e, api_key)[:MAX_ERROR_CHARS]}")

    if st.session_state.poster_raw_image:
        st.image(st.session_state.poster_raw_image, caption="생성된 원본 이미지", width=400)

    if step3:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
        elif not idea.strip():
            st.error("아이디어를 입력해주세요.")
        else:
            with st.spinner("포스터 문구 생성 중..."):
                try:
                    title, period, content = generate_poster_copy(idea, api_key)
                    st.session_state.poster_title = title
                    st.session_state.poster_period = period
                    st.session_state.poster_content = content
                except Exception as e:
                    st.error(f"요청 실패: {redact(e, api_key)[:MAX_ERROR_CHARS]}")

    st.markdown("④ 포스터 문구 (자동 생성됨, 직접 수정 가능 — 안내 포스터 형식: 제목/기간/내용)")
    st.session_state.poster_title = st.text_input(
        "제목 (행사·교육명)", value=st.session_state.poster_title
    )
    tcol1, tcol2 = st.columns([1, 2])
    with tcol1:
        st.session_state.poster_period = st.text_input(
            "기간 (일시)", value=st.session_state.poster_period
        )
    with tcol2:
        st.session_state.poster_content = st.text_input(
            "내용 (장소·대상·신청방법 등)", value=st.session_state.poster_content
        )

    step4 = st.button("4단계: 포스터 합성", type="primary")
    if step4:
        if not st.session_state.poster_raw_image:
            st.error("먼저 2단계에서 이미지를 생성해주세요.")
        else:
            try:
                font_paths = POSTER_FONTS[poster_font_label]
                st.session_state.poster_composed = compose_poster(
                    st.session_state.poster_raw_image,
                    st.session_state.poster_title,
                    st.session_state.poster_period,
                    st.session_state.poster_content,
                    poster_align,
                    poster_position,
                    poster_offset,
                    font_paths,
                    hex_to_rgb(poster_title_color),
                    hex_to_rgb(poster_body_color),
                    poster_shadow_level,
                    poster_border_level,
                    hex_to_rgb(poster_border_color),
                    poster_bar_level,
                    logo_bytes=poster_logo_file.getvalue() if poster_logo_file else None,
                    logo_position=poster_logo_position,
                    logo_size_pct=poster_logo_size,
                    qr_image=generate_qr_image(poster_qr_data) if (poster_qr_enabled and poster_qr_data.strip()) else None,
                    qr_position=poster_qr_position,
                    qr_size_pct=poster_qr_size,
                )
                st.session_state.poster_final = None
            except ValueError as e:
                st.error(str(e))

    if st.session_state.poster_composed:
        st.image(st.session_state.poster_composed, caption="합성된 포스터", width=450)

        step5 = st.button("5단계: 업스케일 적용")
        if step5:
            if poster_upscale == "없음":
                st.session_state.poster_final = st.session_state.poster_composed
            else:
                factor = 2 if poster_upscale == "x2" else 4
                with st.spinner("업스케일 처리 중..."):
                    st.session_state.poster_final = upscale_image(st.session_state.poster_composed, factor)

    if st.session_state.poster_final:
        final = st.session_state.poster_final
        final_img = Image.open(io.BytesIO(final))
        st.markdown(f"⑤ **최종 포스터** ({final_img.width}×{final_img.height})")
        st.image(final, width=500)

        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.download_button("PNG 다운로드", data=final, file_name="poster.png", mime="image/png")
        with dcol2:
            jpg_buf = io.BytesIO()
            final_img.convert("RGB").save(jpg_buf, format="JPEG", quality=95)
            st.download_button("JPG 다운로드", data=jpg_buf.getvalue(), file_name="poster.jpg", mime="image/jpeg")
        with dcol3:
            pdf_buf = io.BytesIO()
            final_img.convert("RGB").save(pdf_buf, format="PDF")
            st.download_button("PDF 다운로드", data=pdf_buf.getvalue(), file_name="poster.pdf", mime="application/pdf")
