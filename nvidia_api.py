"""NVIDIA API 호출부와 모델 설정. Streamlit에 의존하지 않아 app.py·verify_app.py·테스트가 함께 쓴다."""
import base64
import json
import random
import urllib.request

CHAT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
IMAGE_MODELS = {
    # 2026-09-06: 쇼츠·아바타 이미지는 전부 flux.1-dev로 뽑는다(화질 우위). klein은 폴백용.
    "flux.1-dev (고화질, 권장)": {
        "url": "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
        "max_steps": 50, "cfg_scale": 3.5,
    },
    "flux.2-klein-4b (빠름)": {
        "url": "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b",
        "max_steps": 4, "cfg_scale": None,
    },
}
IMAGE_API_URL = IMAGE_MODELS["flux.2-klein-4b (빠름)"]["url"]  # 하위 호환

# 2026-09-05 전면 교체: 이전 등록 모델 7개 중 6개가 410 Gone(EOL)이 되어 사이트가 고장 상태였다.
# 아래는 그날 실제 호출로 생존 + 카테고리별 대표 과제(요약/코드/추론/대화)까지 통과한 모델만 남긴 것.
# 괄호 안 초는 실측 응답시간. NVIDIA는 모델을 자주 EOL 시키므로 오류가 잦아지면 probe_models.py 로 재점검할 것.
MODEL_CATEGORIES = {
    "📝 문서·요약·글쓰기": [
        ("nvidia/ising-calibration-1.5-31b", "빠르고 한국어 요약이 깔끔함 (1.4s)"),
        # minimaxai/minimax-m3 는 2026-09-09 410 Gone → deepseek-v4.1-flash로 교체(2026-09-22 실호출: 요약 정확, 사고과정 때문에 10~14s)
        ("deepseek-ai/deepseek-v4.1-flash", "정확한 한국어 요약 · 사고과정이 있어 10초대 (13.6s)"),
        # 2026-10-02: 9/22엔 타임아웃이던 diffusiongemma가 0.6~2.4s로 정상화(요약·800자 장문 모두 통과).
        # 연속 호출하면 429가 잦고 '숫자만' 같은 초단답은 빈 응답이 나올 수 있다.
        ("google/diffusiongemma-26b-a4b-it", "가장 빠르고 문장이 매끄러움 · 연속 호출 시 429 (0.6~2.4s)"),
        # google/gemma-4-31b-it 는 2026-10-02 실측 47~102s(90초 타임아웃 초과)로 느려져 목록에서 제외
    ],
    "💻 코드·개발": [
        ("openai/gpt-oss-20b", "가볍고 빠른 코드 생성 (1.0s)"),
        ("nvidia/nemotron-3-super-120b-a12b", "대형 모델인데 빠름 · 설명 없이 코드만 (1.5s)"),
        # poolside/laguna-xs-2.1 은 코드 품질은 좋았으나 503이 잦고 응답이 22~39초까지 튀어 제외(2026-09-05)
    ],
    "🧠 심층 추론·복잡한 분석": [
        ("nvidia/nemotron-3-ultra-550b-a55b", "최상위 추론 모델 · 긴 분석에 (7.6s)"),
        ("nvidia/nemotron-3-super-120b-a12b", "추론력 대비 빠름 (1.2s)"),
        ("deepseek-ai/deepseek-v4.1-flash", "지시 준수가 정확한 추론 모델 (2026-09-22 신규, 4~14s)"),
    ],
    "⚡ 빠른 일반대화": [
        ("nvidia/ising-calibration-1.5-31b", "빠르고 자연스러운 한국어 대화"),
        ("google/diffusiongemma-26b-a4b-it", "0.6초대 응답 · 연속 호출 시 429 (2026-10-02 신규)"),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "가볍고 답이 짧음 (2.5s)"),
    ],
}

# 사고과정(reasoning_content)을 답변 대신 뱉을 수 있어 max_tokens를 넉넉히 줘야 하는 모델
REASONING_MODELS = ("openai/gpt-oss", "nvidia/nemotron-3-super", "nvidia/nemotron-3-ultra",
                    "nvidia/nemotron-3.5-lightning", "moonshotai/", "deepseek-ai/")

# 포스터 문구·프롬프트 번역용. 예전엔 meta/llama-3.1-70b-instruct 하나만 썼는데 그게 EOL 되면서
# 포스터 기능이 통째로 죽어 있었다 → 앞에서부터 시도하는 폴백 목록으로 바꿔 한 모델이 죽어도 계속 동작하게.
POSTER_TEXT_MODELS = [
    "nvidia/ising-calibration-1.5-31b",
    "google/diffusiongemma-26b-a4b-it",  # 2026-10-02 추가: 1초 안팎, 429·빈 응답이면 다음 모델로 넘어감
    "deepseek-ai/deepseek-v4.1-flash",   # minimax-m3(410 Gone) 자리, 2026-09-22
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "google/gemma-4-31b-it",             # 2026-10-02 실측 47~102s로 느려져 맨 뒤로
]


MAX_ERROR_CHARS = 300


def redact(text, *secrets):
    """오류 문구에 API 키가 섞여 화면/로그로 나가지 않도록 가린다."""
    text = str(text)
    for sec in secrets:
        if sec:
            text = text.replace(sec, "***")
    return text


def http_error_text(e, api_key=""):
    """HTTPError 본문은 원문 전체 대신 상태코드와 짧게 자른 내용만 보여준다."""
    try:
        body = e.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return f"API 오류 ({e.code}): {redact(body, api_key)[:MAX_ERROR_CHARS]}"


# ---------- NVIDIA API 호출 헬퍼 ----------

def call_chat(model, messages, api_key, temperature=0.7, max_tokens=300):
    payload = {
        "model": model,
        "messages": messages,
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
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        return (message.get("content") or message.get("reasoning_content") or "").strip()


def call_chat_fallback(models, messages, api_key, temperature=0.7, max_tokens=300):
    """models를 앞에서부터 시도. NVIDIA가 모델을 자주 EOL(410)/철거(404) 하고 가끔 503이 나므로
    한 모델이 죽어도 다음 모델로 넘어가 기능 자체는 계속 살아 있게 한다."""
    last = None
    for m in models:
        try:
            out = call_chat(m, messages, api_key, temperature=temperature, max_tokens=max_tokens)
            if out:
                return out
        except Exception as e:  # HTTPError(404/410/503) 및 타임아웃 포함
            last = e
    raise RuntimeError(f"사용 가능한 문구 생성 모델이 없습니다 (마지막 오류: {redact(last, api_key)[:MAX_ERROR_CHARS]})")


def has_hangul(text):
    return any("가" <= ch <= "힣" for ch in text or "")


def translate_to_en(text, api_key):
    """한글 이미지 설명을 영어 프롬프트로. 설명문이 아니라 '프롬프트'로 뽑아야 화질이 산다."""
    messages = [
        {"role": "system", "content": (
            "You turn a Korean description into an English image-generation prompt. "
            "Reply with ONLY the prompt, no quotes, no explanation, no leading label. "
            "Keep it under 700 characters. Use concrete visual nouns and adjectives "
            "(subject, clothing, pose, place, time of day, lighting, camera framing). "
            "Do not invent nudity or brand names.")},
        {"role": "user", "content": text},
    ]
    out = call_chat_fallback(POSTER_TEXT_MODELS, messages, api_key, temperature=0.2, max_tokens=400)
    return (out or "").strip().strip('"').strip()


def call_image(prompt, width, height, steps, api_key, model_key=None):
    # 모델마다 받는 파라미터가 다르다: flux.1-dev는 cfg_scale(9 이하)을 받고,
    # flux.2-klein-4b는 cfg_scale을 보내면 422를 낸다. steps 상한도 다르다.
    spec = IMAGE_MODELS.get(model_key or "flux.2-klein-4b (빠름)")
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": min(int(steps), spec["max_steps"]),
        "seed": random.randint(0, 2**31 - 1),
    }
    if spec["cfg_scale"] is not None:
        payload["cfg_scale"] = spec["cfg_scale"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        spec["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return base64.b64decode(body["artifacts"][0]["base64"])


# ---------- 포스터 문구 생성 ----------

def generate_english_prompt(idea, api_key):
    system = (
        "You are a prompt engineer for an AI image generator (FLUX). "
        "Convert the given Korean poster concept into ONE concise, vivid English visual scene "
        "description (max 50 words) suitable for text-to-image generation. Describe only the "
        "visual scene, subject, mood, lighting, and art style. Do NOT include any text, words, "
        "letters, numbers, or typography in the description — the image itself must contain no "
        "text. Respond with only the prompt, nothing else (no quotes, no explanation)."
    )
    return call_chat_fallback(
        POSTER_TEXT_MODELS,
        [{"role": "system", "content": system}, {"role": "user", "content": idea}],
        api_key,
        temperature=0.6,
        max_tokens=150,
    )


def generate_poster_copy(idea, api_key):
    system = (
        "너는 행사·교육 안내 포스터 카피라이터야. 주어진 내용을 바탕으로 안내 포스터에 들어갈 "
        "문구를 만들어. 반드시 아래 형식 그대로, 라벨을 붙여서 정확히 3줄로만 답해:\n"
        "제목: (행사명/교육명, 15자 이내, 임팩트있게)\n"
        "기간: (일시·기간. 내용에 날짜/요일 정보가 없으면 \"별도 안내\"라고 적어)\n"
        "내용: (장소·대상·신청방법 등 안내사항, 60자 이내, 여러 정보는 ' · '로 구분)"
    )
    raw = call_chat_fallback(
        POSTER_TEXT_MODELS,
        [{"role": "system", "content": system}, {"role": "user", "content": idea}],
        api_key,
        temperature=0.8,
        max_tokens=250,
    )
    title, period, content = "", "", ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("제목:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("기간:"):
            period = line.split(":", 1)[1].strip()
        elif line.startswith("내용:"):
            content = line.split(":", 1)[1].strip()
    return title, period, content
