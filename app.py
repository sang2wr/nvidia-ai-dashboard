import json
import urllib.request
import urllib.error

import streamlit as st

st.set_page_config(page_title="NVIDIA AI 모델 플레이그라운드", page_icon="🟢", layout="wide")

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
    "mistralai/mistral-7b-instruct-v0.3",
    "google/gemma-2-9b-it",
    "microsoft/phi-3-medium-4k-instruct",
]

st.title("🟢 NVIDIA AI 모델 플레이그라운드")

with st.sidebar:
    st.header("설정")

    default_key = st.secrets.get("NVIDIA_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("NVIDIA API Key", value=default_key, type="password")

    model = st.selectbox("모델 선택", MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

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
                API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            full_response = ""
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    full_response = body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                st.error(f"API 오류 ({e.code}): {e.read().decode('utf-8', errors='ignore')}")
                st.stop()
            except Exception as e:
                st.error(f"요청 실패: {e}")
                st.stop()

        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
