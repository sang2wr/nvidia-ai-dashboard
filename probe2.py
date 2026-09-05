"""2026-09-05 재조사: 타임아웃났던 모델 재시험(180초·2회) + 미시험 카탈로그 모델 전수 시험."""
import sys, json, io, re, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open(r'C:\Users\82104\nvidia_ai_dashboard\.streamlit\secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]

RETRY = ['deepseek-ai/deepseek-v4-pro-0813', 'deepseek-ai/deepseek-v4-flash-0731', 'moonshotai/kimi-k3',
         'mistralai/mistral-nemotron', 'meta/llama-3.2-90b-vision-instruct', 'nvidia/nemotron-3-super-120b-a12b']
UNTESTED = ['nvidia/llama-3.1-nemotron-51b-instruct', 'nvidia/nemotron-4-340b-instruct',
            'nvidia/llama3-chatqa-1.5-70b', 'nvidia/mistral-nemo-minitron-8b-8k-instruct',
            'nv-mistralai/mistral-nemo-12b-instruct', 'mistralai/mistral-large-2-instruct',
            'mistralai/mistral-large', 'mistralai/codestral-22b-instruct-v0.1',
            'mistralai/mistral-7b-instruct-v0.3', 'mistralai/mixtral-8x22b-v0.1',
            'google/gemma-3-4b-it', 'google/gemma-2b', 'google/codegemma-7b',
            'ai21labs/jamba-1.5-large-instruct', '01-ai/yi-large', 'databricks/dbrx-instruct',
            'ibm/granite-3.0-8b-instruct', 'ibm/granite-34b-code-instruct', 'bigcode/starcoder2-15b',
            'meta/codellama-70b', 'meta/llama2-70b', 'zyphra/zamba2-7b-instruct',
            'microsoft/phi-3.5-moe-instruct', 'writer/palmyra-med-70b', 'nvidia/nemotron-4-340b-instruct',
            'deepseek-ai/deepseek-coder-6.7b-instruct', 'aisingapore/sea-lion-7b-instruct',
            'nvidia/llama-3.1-nemoguard-8b-content-safety', 'nvidia/nemotron-3.5-content-safety',
            'meta/llama-guard-4-12b', 'nvidia/riva-translate-4b-instruct-v1.1']

def probe(m, timeout=60, tries=1):
    last = ''
    for a in range(tries):
        body = json.dumps({"model": m, "messages": [{"role": "user", "content": "'서울'이라고만 답하세요. 대한민국의 수도는?"}],
                           "max_tokens": 24, "temperature": 0}).encode()
        req = urllib.request.Request('https://integrate.api.nvidia.com/v1/chat/completions', data=body,
              headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json'})
        t = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=timeout)); dt = time.time() - t
            msg = d['choices'][0]['message']
            txt = (msg.get('content') or '') or ('[reasoning]' + (msg.get('reasoning_content') or '')[:60])
            return f"OK   {dt:6.1f}s  {m:50s} {txt.strip()[:64]!r}"
        except urllib.error.HTTPError as e:
            return f"{e.code:<4} {'':8s} {m:50s} {e.read().decode('utf-8','replace')[:100]}"
        except Exception as e:
            last = f"ERR  {'':8s} {m:50s} {type(e).__name__}: {str(e)[:80]}"
            time.sleep(3)
    return last

print("=== 재시험 (180초, 2회) ===", flush=True)
with ThreadPoolExecutor(3) as ex:
    for line in ex.map(lambda m: probe(m, 180, 2), RETRY):
        print(line, flush=True)
print("\n=== 미시험 카탈로그 모델 ===", flush=True)
seen = set()
todo = [m for m in UNTESTED if not (m in seen or seen.add(m))]
with ThreadPoolExecutor(4) as ex:
    for line in ex.map(lambda m: probe(m, 60, 1), todo):
        print(line, flush=True)
