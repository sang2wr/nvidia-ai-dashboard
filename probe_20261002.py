"""2026-10-02 카탈로그 diff + 신규 모델/등록 모델 실호출 점검. urllib 사용(requests는 이 PC에서 멈춤)."""
import json, time, tomllib, urllib.request, urllib.error, sys, re
sys.stdout.reconfigure(encoding='utf-8')
key = tomllib.load(open('.streamlit/secrets.toml', 'rb'))['NVIDIA_API_KEY']
H = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}

cat = json.load(urllib.request.urlopen(urllib.request.Request('https://integrate.api.nvidia.com/v1/models', headers=H), timeout=60))
now = sorted(m['id'] for m in cat['data'])
json.dump(now, open('models_20261002.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
old = json.load(open('models_20260922.json', encoding='utf-8'))
old = sorted(m['id'] if isinstance(m, dict) else m for m in old)
new, gone = [m for m in now if m not in old], [m for m in old if m not in now]
print(f'카탈로그 {len(old)} -> {len(now)}')
print('신규:', new)
print('소멸:', gone)

registered = sorted(set(re.findall(r'"([a-z0-9\-]+/[a-zA-Z0-9.\-]+)"', open('app.py', encoding='utf-8').read())))
registered = [m for m in registered if m in now or m not in new]
PROMPT = '다음을 JSON 배열로만 출력: 경기도 시 3곳의 이름. 설명 없이.'


def probe(m):
    body = json.dumps({'model': m, 'messages': [{'role': 'user', 'content': PROMPT}], 'max_tokens': 300, 'temperature': 0.2}).encode()
    req = urllib.request.Request('https://integrate.api.nvidia.com/v1/chat/completions', data=body, headers=H)
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=90))
        msg = r['choices'][0]['message']
        c = (msg.get('content') or '').strip().replace('\n', ' ')[:90]
        rc = (msg.get('reasoning_content') or '')[:40].replace('\n', ' ')
        print(f'OK   {m:50s} {time.time()-t:5.1f}s content={c!r} reasoning={rc!r}')
    except urllib.error.HTTPError as e:
        print(f'HTTP {e.code} {m:50s} {time.time()-t:5.1f}s {e.read()[:100]!r}')
    except Exception as e:
        print(f'ERR  {m:50s} {time.time()-t:5.1f}s {type(e).__name__} {str(e)[:60]}')
    sys.stdout.flush()


print('\n== 신규 모델 ==')
for m in new:
    probe(m)
print('\n== app.py 등록 모델 ==')
for m in registered:
    if 'flux' not in m:
        probe(m)
