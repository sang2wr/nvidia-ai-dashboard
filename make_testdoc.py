"""정답을 아는 시험용 문서 이미지 생성 (한글 표 + 텍스트)."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont
F = r"C:\Users\82104\nvidia_ai_dashboard\fonts\NanumGothic.ttf"
B = r"C:\Users\82104\nvidia_ai_dashboard\fonts\NanumGothicBold.ttf"
W, H = 1000, 700
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
d.text((40, 30), "2026년 9월 AI 교육 출석부", font=ImageFont.truetype(B, 34), fill="black")
d.text((40, 82), "교육기관: 상상우리   ·   장소: 서울시 마포구 교육장 3층", font=ImageFont.truetype(F, 20), fill="#333")
rows = [["번호", "성명", "생년월일", "출석일수", "수료여부"],
        ["1", "김영수", "1968-03-12", "18", "수료"],
        ["2", "박미정", "1971-11-05", "20", "수료"],
        ["3", "이정호", "1965-07-23", "9",  "미수료"],
        ["4", "최은주", "1973-01-30", "17", "수료"],
        ["5", "정대현", "1969-09-14", "12", "미수료"]]
cw = [90, 180, 240, 150, 180]; x0, y0, rh = 40, 140, 62
fb, fr = ImageFont.truetype(B, 22), ImageFont.truetype(F, 22)
for r, row in enumerate(rows):
    x = x0
    for c, cell in enumerate(row):
        d.rectangle([x, y0 + r*rh, x + cw[c], y0 + (r+1)*rh], outline="#444", width=2,
                    fill="#eaeaea" if r == 0 else "white")
        f = fb if r == 0 else fr
        bb = d.textbbox((0, 0), cell, font=f)
        d.text((x + (cw[c]-bb[2])/2, y0 + r*rh + (rh-bb[3])/2 - 2), cell, font=f, fill="black")
        x += cw[c]
d.text((40, 520), "합계 출석일수: 76일   ·   수료 3명 / 미수료 2명", font=ImageFont.truetype(B, 22), fill="black")
d.text((40, 560), "※ 출석일수 15일 이상인 경우 수료로 인정합니다.", font=ImageFont.truetype(F, 19), fill="#555")
d.text((40, 620), "담당자: 이나윤   연락처: 02-1234-5678", font=ImageFont.truetype(F, 19), fill="#555")
img.save("testdoc.png")
print("saved testdoc.png", os.path.getsize("testdoc.png"), "bytes")
