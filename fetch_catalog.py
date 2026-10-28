import sys, json, io, re, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
key=load_key()
req=urllib.request.Request('https://integrate.api.nvidia.com/v1/models',
    headers={'Authorization':'Bearer '+key,'Accept':'application/json'})
d=json.load(urllib.request.urlopen(req,timeout=60))
ids=sorted(x['id'] for x in d.get('data',[]))
io.open('models_20260904.txt','w',encoding='utf-8').write('\n'.join(ids))
print('TOTAL', len(ids))
