import sys, json, io, re, time, base64, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open(r'.streamlit\secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]
B64 = base64.b64encode(open('testdoc.png','rb').read()).decode()
URL='https://integrate.api.nvidia.com/v1/chat/completions'
def post(p, timeout=120):
    req=urllib.request.Request(URL,data=json.dumps(p).encode(),
      headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r: return json.loads(r.read().decode())
img_html = f'<img src="data:image/png;base64,{B64}" />'
cands = [
 ("img_only + max_tokens 4096", {"model":"nvidia/nemotron-parse","max_tokens":4096,"temperature":0,
    "messages":[{"role":"user","content":img_html}]}),
 ("array image only",           {"model":"nvidia/nemotron-parse","max_tokens":4096,
    "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{B64}"}}]}]}),
 ("img_only + tools markdown",  {"model":"nvidia/nemotron-parse","max_tokens":4096,
    "messages":[{"role":"user","content":img_html}],
    "tools":[{"type":"function","function":{"name":"markdown_bbox"}}]}),
 ("img_only + tools no_bbox",   {"model":"nvidia/nemotron-parse","max_tokens":4096,
    "messages":[{"role":"user","content":img_html}],
    "tools":[{"type":"function","function":{"name":"markdown_no_bbox"}}]}),
 ("img_only + tools detection", {"model":"nvidia/nemotron-parse","max_tokens":4096,
    "messages":[{"role":"user","content":img_html}],
    "tools":[{"type":"function","function":{"name":"detection_only"}}]}),
]
for name,p in cands:
    print("="*15, name, "="*15, flush=True)
    try:
        d=post(p)
        print(json.dumps(d, ensure_ascii=False)[:1800])
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode('utf-8','replace')[:300])
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:120])
    print(flush=True); time.sleep(1)
