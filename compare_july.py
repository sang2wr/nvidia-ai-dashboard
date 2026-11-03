"""2026-07-24 xlsx의 '사용가능 모델' 48개가 2026-09-05에도 살아있는지 전수 재확인."""
import sys, json, io, re, time, urllib.request, urllib.error
from _common import load_key
from concurrent.futures import ThreadPoolExecutor
import os
sys.stdout.reconfigure(encoding='utf-8')
# 사용법: python compare_july.py [xlsx 경로]  (기본: 같은 폴더의 NVIDIA_모델_전수조사.xlsx — 저장소에는 없음)
XLSX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "NVIDIA_모델_전수조사.xlsx")
if not os.path.exists(XLSX):
    sys.exit(f"7월 조사 파일이 없습니다: {XLSX}\n  (저장소에 포함되지 않은 로컬 파일 — 경로를 인자로 넘기세요)")
try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl 이 필요합니다: pip install -r requirements-dev.txt")
key = load_key()
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["사용가능 모델"]
rows = [(r[0], r[1], r[2]) for r in ws.iter_rows(min_row=2, values_only=True) if r[1]]
print(f"7월 '사용가능' {len(rows)}개 재확인\n")
CHAT='https://integrate.api.nvidia.com/v1/chat/completions'
EMB='https://integrate.api.nvidia.com/v1/embeddings'
def probe(item):
    cat, m, desc = item
    isemb = any(k in m for k in ('embed','nvclip','arctic','rerank'))
    url, p = (EMB, {"model":m,"input":["테스트"],"input_type":"query","encoding_format":"float","truncate":"END"}) if isemb else \
             (CHAT, {"model":m,"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0})
    t=time.time()
    try:
        req=urllib.request.Request(url,data=json.dumps(p).encode(),
          headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
        urllib.request.urlopen(req,timeout=60).read()
        return (m, cat, "OK", f"{time.time()-t:.1f}s")
    except urllib.error.HTTPError as e:
        body=e.read().decode('utf-8','replace')
        tag = "410 EOL" if e.code==410 else ("404 없음" if e.code==404 else f"{e.code}")
        return (m, cat, tag, body[:60].replace("\n"," "))
    except Exception as e:
        return (m, cat, "타임아웃" if 'imeout' in type(e).__name__ else type(e).__name__, "")
with ThreadPoolExecutor(5) as ex:
    res = list(ex.map(probe, rows))
alive=[r for r in res if r[2]=="OK"]
for m,cat,st,extra in sorted(res, key=lambda x:(x[2]!="OK", x[0])):
    print(f"  {'✅' if st=='OK' else '❌'} {st:9s} {m:52s} {(cat or '')[:26]}")
print(f"\n7월 사용가능 {len(rows)}개 → 9월 생존 {len(alive)}개 ({len(alive)*100//max(1,len(rows))}%)")
from collections import Counter
print("사유:", dict(Counter(r[2] for r in res)))
