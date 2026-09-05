"""카테고리 적합도 시험: 후보 모델에 '문서/코드/추론/빠른대화' 대표 과제를 실제로 던져 본다.
probe2.py가 생사만 본다면, 이건 대시보드 어느 칸에 넣을지 정하기 위한 시험.
"""
import sys, json, io, re, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open(r'C:\Users\82104\nvidia_ai_dashboard\.streamlit\secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]

TASKS = {
 "문서": ("다음을 한국어 두 문장으로 요약하세요. 요약문만 출력:\n"
          "상상우리는 중장년 인재를 사회적경제 조직과 연결하는 사업을 운영한다. "
          "2026년에는 AX 현장컨설팅을 도입해 기업의 디지털 전환을 지원했고, 참여 기업 만족도가 높았다.", 160),
 "코드": ("파이썬 함수만 출력(설명 금지): 문자열 리스트를 받아 길이 내림차순으로 정렬해 반환하는 함수 sort_by_len", 160),
 "추론": ("한 상자에 사과가 12개씩 든 상자가 7개 있고 3개를 먹었다. 남은 사과 수를 구하고 "
          "마지막 줄에 '답: N' 형식으로만 결론을 쓰시오.", 300),
 "대화": ("친구에게 하듯 한 문장으로 답하세요. 오늘 기분이 어때?", 80),
}

def ask(model, prompt, mx):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "temperature": 0.3}).encode()
    req = urllib.request.Request('https://integrate.api.nvidia.com/v1/chat/completions', data=body,
          headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'Accept': 'application/json'})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120)); dt = time.time() - t
        m = d['choices'][0]['message']
        c, r = (m.get('content') or '').strip(), (m.get('reasoning_content') or '').strip()
        return dt, c, r
    except urllib.error.HTTPError as e:
        return -1, f"HTTP {e.code}", ''
    except Exception as e:
        return -1, f"{type(e).__name__}", ''

def run(model):
    lines = [f"\n##### {model}"]
    for name, (p, mx) in TASKS.items():
        dt, c, r = ask(model, p, mx)
        flag = "content" if c else ("reasoning만" if r else "빈응답")
        body = (c or r).replace("\n", " ⏎ ")[:150]
        lines.append(f"  [{name}] {dt:5.1f}s {flag:9s} {body}")
        time.sleep(0.5)
    return "\n".join(lines)

MODELS = sys.argv[1:] or [
 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', 'minimaxai/minimax-m3',
 'poolside/laguna-xs-2.1', 'nvidia/ising-calibration-1.5-31b', 'google/gemma-4-31b-it',
 'nvidia/nemotron-3-ultra-550b-a55b', 'openai/gpt-oss-20b']
with ThreadPoolExecutor(3) as ex:
    for line in ex.map(run, MODELS):
        print(line, flush=True)
