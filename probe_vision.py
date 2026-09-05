"""nemotron-parse(문서 파서) + 비전 모델들에 같은 문서 이미지를 넣어 정확도를 본다.
NVIDIA VLM은 모델마다 입력 형식이 달라 3가지 형식을 순서대로 시도한다."""
import sys, json, io, re, time, base64, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open(r'C:\Users\82104\nvidia_ai_dashboard\.streamlit\secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]
B64 = base64.b64encode(open('testdoc.png', 'rb').read()).decode()
PROMPT = "이 문서의 표를 그대로 읽어서 markdown 표로 옮기고, 아래 안내 문구도 그대로 적어줘."

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def forms(model):
    """(형식이름, payload) 후보들"""
    img_html = f'<img src="data:image/png;base64,{B64}" />'
    return [
        ("content_array", {"model": model, "max_tokens": 1200, "temperature": 0, "messages": [
            {"role": "user", "content": [{"type": "text", "text": PROMPT},
                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}}]}]}),
        ("img_tag", {"model": model, "max_tokens": 1200, "temperature": 0, "messages": [
            {"role": "user", "content": PROMPT + " " + img_html}]}),
        ("img_only", {"model": model, "max_tokens": 1200, "temperature": 0, "messages": [
            {"role": "user", "content": img_html}]}),
    ]

MODELS = sys.argv[1:] or ['nvidia/nemotron-parse']
URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
for m in MODELS:
    print(f"\n{'='*20} {m} {'='*20}", flush=True)
    for name, payload in forms(m):
        t = time.time()
        try:
            d = post(URL, payload)
            msg = d['choices'][0]['message']
            out = (msg.get('content') or msg.get('reasoning_content') or '')
            print(f"[{name}] ✅ {time.time()-t:.1f}s len={len(out)}")
            print(out[:1400])
            break                      # 통하는 형식 하나 찾으면 끝
        except urllib.error.HTTPError as e:
            print(f"[{name}] {e.code} {e.read().decode('utf-8','replace')[:160]}")
        except Exception as e:
            print(f"[{name}] ERR {type(e).__name__}: {str(e)[:100]}")
        time.sleep(1)
