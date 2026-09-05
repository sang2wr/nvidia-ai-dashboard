"""nemotron-parse 정확도 확인 + 비전/임베딩 계열 전수 시험 (2026-09-05)."""
import sys, json, io, re, time, base64, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open(r'.streamlit\secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]
B64 = base64.b64encode(open('testdoc.png','rb').read()).decode()
IMG = f'<img src="data:image/png;base64,{B64}" />'
CHAT = 'https://integrate.api.nvidia.com/v1/chat/completions'
EMB  = 'https://integrate.api.nvidia.com/v1/embeddings'

def post(url, p, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(p).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# ---------- 1) nemotron-parse: markdown_no_bbox 로 깨끗한 마크다운 뽑기 ----------
print("="*24, "nemotron-parse (markdown_no_bbox)", "="*24)
def parse_doc(b64):
    d = post(CHAT, {"model":"nvidia/nemotron-parse","max_tokens":4096,
        "messages":[{"role":"user","content":f'<img src="data:image/png;base64,{b64}" />'}],
        "tools":[{"type":"function","function":{"name":"markdown_no_bbox"}}]})
    args = d['choices'][0]['message']['tool_calls'][0]['function']['arguments']
    return "\n".join(x.get('text','') for x in json.loads(args))
t=time.time(); md = parse_doc(B64)
print(f"({time.time()-t:.1f}s, {len(md)}자)\n{md}\n")

TRUTH = {"김영수":"1968-03-12|18|수료","박미정":"1971-11-05|20|수료","이정호":"1965-07-23|9|미수료",
         "최은주":"1973-01-30|17|수료","정대현":"1969-09-14|12|미수료"}
flat = md.replace(" ","")
print("--- 표 정확도 ---")
for name,v in TRUTH.items():
    ok = all(tok.replace(" ","") in flat for tok in [name]+v.split("|"))
    print(f"  {'✅' if ok else '❌'} {name} {v}")
for k in ["2026년9월AI교육출석부","상상우리","서울시마포구교육장3층","76일","15일이상","02-1234-5678","이나윤"]:
    print(f"  {'✅' if k in flat else '❌'} {k}")

# ---------- 2) 비전 모델 ----------
print("\n" + "="*24, "비전(VLM) 모델", "="*24)
VIS = ['meta/llama-3.2-11b-vision-instruct','microsoft/phi-3-vision-128k-instruct','nvidia/neva-22b',
       'nvidia/vila','microsoft/kosmos-2','google/deplot','adept/fuyu-8b','nvidia/cosmos-reason2-8b',
       'nvidia/ai-synthetic-video-detector','nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1']
Q = "이 문서의 제목과 표의 행 수를 알려줘."
for m in VIS:
    for form,payload in [
      ("array",{"model":m,"max_tokens":300,"temperature":0,"messages":[{"role":"user","content":[
          {"type":"text","text":Q},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{B64}"}}]}]}),
      ("imgtag",{"model":m,"max_tokens":300,"temperature":0,"messages":[{"role":"user","content":Q+" "+IMG}]}),
    ]:
        t=time.time()
        try:
            d=post(CHAT,payload); msg=d['choices'][0]['message']
            out=(msg.get('content') or msg.get('reasoning_content') or '')
            if not out and msg.get('tool_calls'):
                out='[tool_calls] '+str(msg['tool_calls'][0]['function']['arguments'])[:120]
            print(f"  ✅ {time.time()-t:5.1f}s {m:46s} [{form}] {out.strip()[:90]!r}"); break
        except urllib.error.HTTPError as e:
            err=e.read().decode('utf-8','replace')[:70]
            if form=="imgtag": print(f"  ❌ {'':5s} {m:46s} {e.code} {err}")
        except Exception as e:
            if form=="imgtag": print(f"  ❌ {'':5s} {m:46s} {type(e).__name__}")
        time.sleep(0.7)

# ---------- 3) 임베딩 모델 ----------
print("\n" + "="*24, "임베딩 모델", "="*24)
EMBM = ['nvidia/nemotron-3-embed-1b','nvidia/llama-3.2-nv-embedqa-1b-v1','nvidia/embed-qa-4',
        'nvidia/nv-embedqa-mistral-7b-v2','snowflake/arctic-embed-l','nvidia/nvclip',
        'nvidia/llama-nemotron-embed-vl-1b-v2']
for m in EMBM:
    t=time.time()
    try:
        d=post(EMB,{"model":m,"input":["중장년 인재 채용 공고"],"input_type":"query",
                    "encoding_format":"float","truncate":"END"})
        v=d['data'][0]['embedding']
        print(f"  ✅ {time.time()-t:5.1f}s {m:44s} dim={len(v)}")
    except urllib.error.HTTPError as e:
        print(f"  ❌ {'':5s} {m:44s} {e.code} {e.read().decode('utf-8','replace')[:80]}")
    except Exception as e:
        print(f"  ❌ {'':5s} {m:44s} {type(e).__name__}")
    time.sleep(0.7)
