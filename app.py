import base64
import json
import random
import urllib.request
import urllib.error

import streamlit as st

st.set_page_config(page_title="NVIDIA AI 모델 플레이그라운드", page_icon="🟢", layout="wide")

CHAT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
IMAGE_API_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"

MODEL_CATEGORIES = {
    "📝 문서·요약·글쓰기": [
        ("meta/llama-3.3-70b-instruct", "최신 대형 모델 · 정교한 글쓰기/요약에 강함"),
        ("meta/llama-3.1-70b-instruct", "안정적인 문서 작성·분석"),
    ],
    "💻 코드·개발": [
        ("openai/gpt-oss-20b", "가볍고 빠른 코드 생성"),
        ("openai/gpt-oss-120b", "복잡한 코드·디버깅에 강함(느림)"),
    ],
    "🧠 심층 추론·복잡한 분석": [
        ("nvidia/llama-3.3-nemotron-super-49b-v1", "NVIDIA 튜닝 · 복잡한 추론/분석 특화"),
    ],
    "⚡ 빠른 일반대화": [
        ("meta/llama-3.1-8b-instruct", "가장 빠른 응답 · 일상 대화"),
        ("mistralai/mixtral-8x7b-instruct-v0.1", "MoE 구조 · 빠르면서 균형잡힌 성능"),
    ],
}

IMAGE_SIZES = {
    "정사각형 (1024×1024)": (1024, 1024),
    "세로형 (832×1216)": (832, 1216),
    "가로형 (1216×832)": (1216, 832),
}

st.title("🟢 NVIDIA AI 모델 플레이그라운드")

with st.sidebar:
    st.header("설정")

    default_key = st.secrets.get("NVIDIA_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("NVIDIA API Key", value=default_key, type="password")

    mode = st.radio("모드", ["💬 텍스트 채팅", "🎨 이미지 생성"])

    st.divider()

    if mode == "💬 텍스트 채팅":
        category = st.selectbox("작업 유형", list(MODEL_CATEGORIES.keys()))
        model_options = MODEL_CATEGORIES[category]
        model_labels = [f"{name}  —  {desc}" for name, desc in model_options]
        selected_label = st.selectbox("모델 선택", model_labels)
        model = model_options[model_labels.index(selected_label)][0]

        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max tokens", 64, 4096, 768, 64)
        if model.startswith("openai/gpt-oss"):
            st.caption("⚠️ 추론(reasoning) 모델이라 max_tokens가 낮으면 답변이 잘릴 수 있습니다.")

        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("모델: black-forest-labs/flux.1-dev")
        size_label = st.selectbox("이미지 비율", list(IMAGE_SIZES.keys()))
        steps = st.slider("Steps (품질/속도)", 10, 50, 25, 5)
        cfg_scale = st.slider("CFG Scale (프롬프트 반영 강도)", 1.0, 10.0, 5.0, 0.5)
        st.caption("⏱️ 이미지 1장 생성에 15~30초 정도 걸립니다.")

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

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                payload = {
                    "model": model,
                    "messages": st.session_state.messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                req = urllib.request.Request(
                    CHAT_API_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )

                full_response = ""
                try:
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                        message = body["choices"][0]["message"]
                        full_response = message.get("content") or message.get("reasoning_content") or ""
                except urllib.error.HTTPError as e:
                    st.error(f"API 오류 ({e.code}): {e.read().decode('utf-8', errors='ignore')}")
                    st.stop()
                except Exception as e:
                    st.error(f"요청 실패: {e}")
                    st.stop()

            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    if "images" not in st.session_state:
        st.session_state.images = []

    prompt = st.text_area(
        "이미지 설명 (프롬프트, 영어로 작성하면 품질이 더 좋습니다)",
        placeholder="예: a cute orange cat sitting on a windowsill, digital art",
        height=100,
    )
    generate = st.button("🎨 이미지 생성", type="primary")

    if generate:
        if not api_key:
            st.error("사이드바에 NVIDIA API Key를 입력해주세요.")
            st.stop()
        if not prompt.strip():
            st.error("이미지 설명을 입력해주세요.")
            st.stop()

        width, height = IMAGE_SIZES[size_label]
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "seed": random.randint(0, 2**31 - 1),
            "mode": "base",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            IMAGE_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with st.spinner("이미지 생성 중... (15~30초 소요)"):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    img_bytes = base64.b64decode(body["artifacts"][0]["base64"])
                    st.session_state.images.insert(0, {"prompt": prompt, "bytes": img_bytes})
            except urllib.error.HTTPError as e:
                st.error(f"API 오류 ({e.code}): {e.read().decode('utf-8', errors='ignore')}")
            except Exception as e:
                st.error(f"요청 실패: {e}")

    for i, item in enumerate(st.session_state.images):
        st.image(item["bytes"], caption=item["prompt"], width=512)
        st.download_button(
            "다운로드",
            data=item["bytes"],
            file_name=f"nvidia_flux_{i}.png",
            mime="image/png",
            key=f"dl_{i}",
        )
        st.divider()
