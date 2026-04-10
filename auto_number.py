"""
운빨존많겜 시참 코드 자동 입력 프로그램

F7  - 초대 코드 위치 등록 (숫자 위에 마우스 올리고 누르기)
F2  - [테스트] 숫자 인식 + 클립보드 저장
F3  - [테스트] 입력창 탐색
F4  - [테스트] 참가 버튼 탐색
F9  - 자동 대기 시작/중지
ESC - 종료
"""

import os
import io
import json
import time
import threading
import numpy as np
import cv2
import mss
import pyautogui
import ddddocr
import keyboard
import pyperclip
from PIL import Image

ocr = ddddocr.DdddOcr(show_ad=False)

DEBUG_DIR      = "debug"
CONFIG_FILE    = "config.json"
TEMPLATE_INPUT = os.path.join("images", "input.png")
TEMPLATE_JOIN  = os.path.join("images", "join.png")

SCALES = [round(s, 2) for s in np.arange(0.5, 1.6, 0.2)]
MATCH_THRESHOLD = 0.6

# 초대 코드 고정 캡처 위치 (F7로 설정)
capture_region = None  # (x, y, w, h)

# 마지막 발견 좌표 캐시
_cache = {}  # template_path → (x, y, w, h, scale)


# ── 화면 캡처 ────────────────────────────────────────────

def grab_screen(region=None) -> np.ndarray:
    with mss.mss() as sct:
        if region:
            x, y, w, h = region
            mon = {"left": x, "top": y, "width": w, "height": h}
        else:
            mon = sct.monitors[1]
        return np.array(sct.grab(mon))  # BGRA


def grab_pil(region=None) -> Image.Image:
    arr = grab_screen(region)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB))


# ── 설정 저장/로드 ────────────────────────────────────────

def load_config():
    global capture_region
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        capture_region = data.get("capture_region")
        if capture_region:
            print(f"[설정 로드] 캡처 위치: {capture_region}")


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({"capture_region": capture_region}, f, indent=2)


# ── 템플릿 매칭 (입력창/참가 버튼용) ─────────────────────

def find_on_screen(template_path: str, screen_gray: np.ndarray):
    buf = np.fromfile(template_path, dtype=np.uint8)
    tmpl_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if tmpl_bgr is None:
        print(f"[오류] 템플릿 없음: {template_path}")
        return None

    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_edge = cv2.Canny(tmpl_gray, 50, 150)
    screen_edge = cv2.Canny(screen_gray, 50, 150)
    th, tw = tmpl_edge.shape[:2]
    best_val, best_loc, best_scale = 0, None, 1.0

    for scale in SCALES:
        nw, nh = int(tw * scale), int(th * scale)
        if nw < 10 or nh < 10:
            continue
        resized = cv2.resize(tmpl_edge, (nw, nh))
        if resized.shape[0] > screen_edge.shape[0] or resized.shape[1] > screen_edge.shape[1]:
            continue
        result = cv2.matchTemplate(screen_edge, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val, best_loc, best_scale = max_val, max_loc, scale

    if best_val >= MATCH_THRESHOLD and best_loc:
        nw, nh = int(tw * best_scale), int(th * best_scale)
        x, y = best_loc
        print(f"  [{os.path.basename(template_path)}] 발견 (신뢰도: {best_val:.2f}) → ({x+nw//2}, {y+nh//2})")
        # 디버그: 전체 화면에 매칭 위치 표시
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_img = cv2.cvtColor(screen_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_img, (x, y), (x+nw, y+nh), (0, 0, 255), 3)
        cv2.circle(debug_img, (x+nw//2, y+nh//2), 8, (0, 255, 0), -1)
        name = os.path.splitext(os.path.basename(template_path))[0]
        cv2.imwrite(os.path.join(DEBUG_DIR, f"match_{name}.png"), debug_img)
        return (x, y, nw, nh, best_scale)

    print(f"  [{os.path.basename(template_path)}] 미발견 (신뢰도: {best_val:.2f})")
    return None


# ── 초대 코드 OCR ────────────────────────────────────────

def ocr_number() -> str:
    if not capture_region:
        print("[오류] F7로 초대 코드 위치를 먼저 설정하세요.")
        return ""

    img = grab_pil(region=capture_region)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    img.save(os.path.join(DEBUG_DIR, "raw.png"))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    text = ocr.classification(buf.getvalue())
    # 흔한 오인식 보정
    text = text.replace('o', '0').replace('O', '0') \
               .replace('l', '1').replace('I', '1') \
               .replace('s', '5').replace('S', '5') \
               .replace('g', '9').replace('q', '9') \
               .replace('b', '6').replace('B', '8')
    digits = "".join(filter(str.isdigit, text))
    print(f"  [OCR] '{text}' → '{digits}'")
    return digits


# ── F7: 위치 저장 ─────────────────────────────────────────

def save_capture_pos():
    global capture_region
    cx, cy = pyautogui.position()

    scan = 200
    arr = grab_screen(region=(cx - scan, cy - scan, scan * 2, scan * 2))
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2HSV)

    # 어두운 색 마스크 (초대 코드 박스 배경: 어두운 갈색/검정)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 마우스 위치 포함 여부
        if not (x <= scan <= x + w and y <= scan <= y + h):
            continue
        # 가로가 세로보다 긴 직사각형 (캐릭터 아이콘 제외)
        if w <= h:
            continue
        # 크기 점수 (적당히 큰 것 우선)
        score = w * h
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best:
        bx, by, bw, bh = best
        pad = 4
        abs_x = (cx - scan) + bx - pad
        abs_y = (cy - scan) + by - pad
        capture_region = (abs_x, abs_y, bw + pad * 2, bh + pad * 2)
        print(f"[F7] 박스 감지: {bw}x{bh} → 저장 크기 {capture_region[2]}x{capture_region[3]}")
    else:
        capture_region = (cx - 55, cy - 25, 110, 50)
        print(f"[F7] 감지 실패 → 기본값 사용")

    save_config()
    print(f"[F7 완료] 위치 저장 완료: {capture_region[2]}x{capture_region[3]} at ({capture_region[0]},{capture_region[1]})\n")


# ── 메인 실행 ────────────────────────────────────────────

_running = False


def run_auto_loop():
    global _running
    _running = True
    print("[F9] 자동 대기 시작... (F9 다시 누르면 중지)\n")

    while _running:
        t0 = time.time()
        screen_gray = cv2.cvtColor(grab_screen(), cv2.COLOR_BGRA2GRAY)
        number_result, inp_result, jn_result = [None], [None], [None]

        def do_ocr():
            number_result[0] = ocr_number()

        def find_inp():
            inp_result[0] = find_on_screen(TEMPLATE_INPUT, screen_gray)

        def find_jn():
            jn_result[0] = find_on_screen(TEMPLATE_JOIN, screen_gray)

        t_ocr = threading.Thread(target=do_ocr)
        t_inp = threading.Thread(target=find_inp)
        t_jn  = threading.Thread(target=find_jn)
        t_ocr.start(); t_inp.start(); t_jn.start()
        t_ocr.join();  t_inp.join();  t_jn.join()
        t1 = time.time()
        print(f"  OCR+탐색: {t1-t0:.2f}s")

        number = number_result[0]
        if not number or len(number) != 4:
            time.sleep(1)
            continue

        if not inp_result[0]:
            time.sleep(1)
            continue

        inp = inp_result[0]
        pyautogui.click(inp[0] + inp[2]//2, inp[1] + inp[3]//2)
        time.sleep(0.05)
        print(f"[인식된 숫자] {number}")
        for digit in number:
            keyboard.press_and_release(digit)
        pyautogui.press("enter")
        t2 = time.time()
        print(f"  입력: {t2-t1:.2f}s")

        if jn_result[0]:
            jn = jn_result[0]
            pyautogui.click(jn[0] + jn[2] // 2, jn[1] + jn[3] // 2)

        t4 = time.time()
        print(f"[완료] '{number}' 총 {t4-t0:.2f}s / 종료\n")
        os._exit(0)

    print("[중지] 자동 대기 종료\n")


# ── 테스트 ───────────────────────────────────────────────

def test_number_ocr():
    print("[F2 테스트] 초대 코드 숫자 인식 중...")
    number = ocr_number()
    if number:
        pyperclip.copy(number)
        print(f"[성공] 인식된 숫자: {number}  ← 클립보드 복사됨\n")
    else:
        print("[실패] 숫자를 인식하지 못했습니다.\n")


def test_input_field():
    print("[F3 테스트] 입력창 탐색 중...")
    screen_gray = cv2.cvtColor(grab_screen(), cv2.COLOR_BGRA2GRAY)
    found = find_on_screen(TEMPLATE_INPUT, screen_gray)
    if found:
        print(f"[성공] 입력창 발견: ({found[0] + found[2]//2}, {found[1] + found[3]//2})\n")
    else:
        print("[실패] 입력창을 찾지 못했습니다.\n")


def test_join_button():
    print("[F4 테스트] 참가 버튼 탐색 중...")
    screen_gray = cv2.cvtColor(grab_screen(), cv2.COLOR_BGRA2GRAY)
    found = find_on_screen(TEMPLATE_JOIN, screen_gray)
    if found:
        print(f"[성공] 참가 버튼 발견: ({found[0] + found[2]//2}, {found[1] + found[3]//2})\n")
    else:
        print("[실패] 참가 버튼을 찾지 못했습니다.\n")


def on_f2(): threading.Thread(target=test_number_ocr,  daemon=True).start()
def on_f3(): threading.Thread(target=test_input_field, daemon=True).start()
def on_f4(): threading.Thread(target=test_join_button, daemon=True).start()
def on_f7(): threading.Thread(target=save_capture_pos, daemon=True).start()

def on_f9():
    global _running
    if _running:
        _running = False
    else:
        threading.Thread(target=run_auto_loop, daemon=True).start()


def main():
    print("=" * 45)
    print("  운빨존많겜 시참 코드 자동 입력 프로그램")
    print("=" * 45)
    print("  F7  - 초대 코드 위치 등록 (숫자 위에 마우스)")
    print("  F2  - 숫자 인식 테스트")
    print("  F3  - 입력창 탐색 테스트")
    print("  F4  - 참가 버튼 탐색 테스트")
    print("  F9  - 자동 대기 시작 / 중지")
    print("  ESC - 종료")
    print("=" * 45 + "\n")

    load_config()

    keyboard.add_hotkey("F2", on_f2)
    keyboard.add_hotkey("F3", on_f3)
    keyboard.add_hotkey("F4", on_f4)
    keyboard.add_hotkey("F7", on_f7)
    keyboard.add_hotkey("F9", on_f9)

    try:
        keyboard.wait("esc")
    except KeyboardInterrupt:
        pass
    print("종료합니다.")


if __name__ == "__main__":
    main()
