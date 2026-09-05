"""app.py의 모델 설정이 실제로 동작하는지 Streamlit 없이 직접 검증."""
import sys, io, re, types, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\82104\nvidia_ai_dashboard')

# streamlit/qrcode 없이 app.py를 import 하기 위한 최소 스텁
class _Stub:
    def __getattr__(self, n): return _Stub()
    def __call__(self, *a, **k): return _Stub()
    def __enter__(self): return _Stub()
    def __exit__(self, *a): return False
for name in ('streamlit', 'qrcode'):
    sys.modules[name] = _Stub()

src = io.open('app.py', encoding='utf-8').read()
head = src.split('# ---------- 포스터 합성 ----------')[0]
mod = types.ModuleType('appcore'); mod.__dict__['__file__'] = 'app.py'
exec(compile(head, 'app.py', 'exec'), mod.__dict__)

key = [re.match(r'\s*NVIDIA_API_KEY\s*=\s*["\'](.+?)["\']', l).group(1)
       for l in io.open('.streamlit/secrets.toml', encoding='utf-8')
       if re.match(r'\s*NVIDIA_API_KEY\s*=', l)][0]

print("=== 1) 채팅 카테고리별 등록 모델 전수 호출 ===")
ok = fail = 0
for cat, models in mod.MODEL_CATEGORIES.items():
    print(f"\n[{cat}]")
    for m, desc in models:
        t = time.time()
        try:
            out = mod.call_chat(m, [{"role": "user", "content": "한 문장으로 자기소개 해줘."}],
                                key, temperature=0.5, max_tokens=200)
            print(f"  ✅ {time.time()-t:5.1f}s {m:48s} {out[:56]!r}"); ok += 1
        except Exception as e:
            print(f"  ❌ {'':5s} {m:48s} {type(e).__name__}: {str(e)[:60]}"); fail += 1
        time.sleep(0.8)

print(f"\n=== 2) 포스터 문구 생성 (폴백 목록) ===")
try:
    t = time.time()
    en = mod.generate_english_prompt("가을 중장년 AI 교육 수강생 모집", key)
    print(f"  ✅ 영문 프롬프트 {time.time()-t:.1f}s: {en[:110]}")
except Exception as e:
    print(f"  ❌ 영문 프롬프트: {e}"); fail += 1
try:
    t = time.time()
    ttl, per, con = mod.generate_poster_copy("10월 중장년 AI 교육, 서울시 주최, 선착순 30명", key)
    print(f"  ✅ 문구 {time.time()-t:.1f}s → 제목={ttl!r} 기간={per!r} 내용={con[:50]!r}")
    if not ttl: print("  ⚠️ 제목이 비었음 — 형식 파싱 확인 필요"); fail += 1
except Exception as e:
    print(f"  ❌ 문구: {e}"); fail += 1

print(f"\n=== 3) 이미지 생성 (flux.2-klein-4b) ===")
try:
    t = time.time()
    b = mod.call_image("autumn classroom, warm light, no text", 832, 1216, 4, key)
    print(f"  ✅ {time.time()-t:.1f}s · {len(b):,} bytes")
except Exception as e:
    print(f"  ❌ {e}"); fail += 1

print(f"\n결과: 실패 {fail}건")
