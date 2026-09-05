import sys, json, io, re, time, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
key=[re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']',l).group(1)
     for l in io.open(r'C:\Users\82104\nvidia_ai_dashboard\.streamlit\secrets.toml',encoding='utf-8')
     if re.match(r'\s*NVIDIA_API_KEY\s*=',l)][0]

CUR=['meta/llama-3.3-70b-instruct','meta/llama-3.1-70b-instruct','meta/llama-3.1-8b-instruct',
     'openai/gpt-oss-120b','openai/gpt-oss-20b','nvidia/llama-3.3-nemotron-super-49b-v1.5','z-ai/glm-5.2']
NEW=['nvidia/nemotron-3-ultra-550b-a55b','nvidia/nemotron-3-super-120b-a12b','nvidia/nemotron-nano-3-30b-a3b',
     'nvidia/nemotron-3.5-lightning-30b-a3b','nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
     'deepseek-ai/deepseek-v4-pro-0813','deepseek-ai/deepseek-v4-flash-0731','moonshotai/kimi-k3','moonshotai/kimi-k2.6',
     'minimaxai/minimax-m3','google/gemma-4-31b-it','google/gemma-3-12b-it','mistralai/mistral-nemotron',
     'nvidia/llama-3.1-nemotron-70b-instruct','nvidia/llama-3.1-nemotron-ultra-253b-v1',
     'writer/palmyra-fin-70b-32k','writer/palmyra-creative-122b','nvidia/riva-translate-4b-instruct-v2',
     'meta/llama-3.2-90b-vision-instruct','nvidia/cosmos-reason2-8b','nvidia/nemotron-parse',
     'poolside/laguna-xs-2.1','meta/muse-glimmer-30b','nvidia/ising-calibration-1.5-31b','google/diffusiongemma-26b-a4b-it']

def probe(m):
    body=json.dumps({"model":m,"messages":[{"role":"user","content":"'서울'이라고만 답하세요. 대한민국의 수도는?"}],
                     "max_tokens":24,"temperature":0}).encode()
    req=urllib.request.Request('https://integrate.api.nvidia.com/v1/chat/completions',data=body,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
    t=time.time()
    try:
        d=json.load(urllib.request.urlopen(req,timeout=90)); dt=time.time()-t
        msg=d['choices'][0]['message']
        txt=(msg.get('content') or '') or ('[reasoning]'+(msg.get('reasoning_content') or '')[:60])
        return f"OK   {dt:5.1f}s  {m:52s} {txt.strip()[:70]!r}"
    except urllib.error.HTTPError as e:
        return f"{e.code:<4} {'':7s} {m:52s} {e.read().decode('utf-8','replace')[:110]}"
    except Exception as e:
        return f"ERR  {'':7s} {m:52s} {type(e).__name__}: {str(e)[:90]}"

for label,lst in (('=== 현재 대시보드가 쓰는 모델 ===',CUR),('\n=== 신규 후보 ===',NEW)):
    print(label, flush=True)
    for m in lst:
        print(probe(m), flush=True); time.sleep(1.7)
