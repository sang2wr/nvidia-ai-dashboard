import sys, json, io, re, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
key=None
for line in io.open(r'C:\Users\82104\nvidia_ai_dashboard\.streamlit\secrets.toml',encoding='utf-8'):
    m=re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']',line)
    if m: key=m.group(1); break
if not key: print('KEY_NOT_FOUND'); sys.exit(1)
req=urllib.request.Request('https://integrate.api.nvidia.com/v1/models',
    headers={'Authorization':'Bearer '+key,'Accept':'application/json'})
d=json.load(urllib.request.urlopen(req,timeout=60))
ids=sorted(x['id'] for x in d.get('data',[]))
io.open(r'C:\Users\82104\AppData\Local\Temp\claude\C--Users-82104\d432b839-afa7-47fc-a9b3-86b68b1aa82c\scratchpad\models_20260904.txt','w',encoding='utf-8').write('\n'.join(ids))
print('TOTAL', len(ids))
