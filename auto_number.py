"""
유튜브 시참 번호 자동 입력 프로그램

F9  - 실행 (초대 코드 자동 탐색 → OCR → 입력)
ESC - 종료
"""

import os
import time
import threading
import numpy as np
import cv2
import pyautogui
import ddddocr
import keyboard
import pyperclip
from PIL import ImageGrab, Image
import pynput.mouse as pmouse

ocr = ddddocr.DdddOcr(show_ad=False)

TEMPLATE_DIR    = "images"
DEBUG_DIR       = "debug"
TEMPLATE_HEADER = os.path.join(TEMPLATE_DIR, "number_header.png")  # "초대 코드" 텍스트만
TEMPLATE_INPUT  = os.path.join(TEMPLATE_DIR, "input.png")
TEMPLATE_JOIN   = os.path.join(TEMPLATE_DIR, "join.png")

# number.png 전체 높이 대비 header 비율 (숫자 영역 계산용)
HEADER_RATIO = 0.42   # number_header = number.png 상단 42%

SCALES = [round(s, 2) for s in np.arange(0.4, 2.2, 0.05)]
MATCH_THRESHOLD = 0.55


# ── 멀티스케일 템플릿 매칭 ───────────────────────────────

def find_on_screen(template_path: str, screen_gray: np.ndarray):
    """화면(gray)에서 template를 찾아 (x, y, w, h, scale) 반환. 없으면 None."""
    # cv2.imread는 Windows에서 한글 경로를 못 읽으므로 np.fromfile 사용
    buf = np.fromfile(template_path, dtype=np.uint8)
    tmpl_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if tmpl_bgr is None:
        print(f"[오류] 템플릿 없음: {template_path}")
        return None

    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl_gray.shape[:2]

    best_val, best_loc, best_scale = 0, None, 1.0

    for scale in SCALES:
        nw, nh = int(tw * scale), int(th * scale)
        if nw < 10 or nh < 10:
            continue
        resized = cv2.resize(tmpl_gray, (nw, nh))
        if resized.shape[0] > screen_gray.shape[0] or resized.shape[1] > screen_gray.shape[1]:
            continue
        result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val, best_loc, best_scale = max_val, max_loc, scale

    if best_val >= MATCH_THRESHOLD and best_loc:
        nw, nh = int(tw * best_scale), int(th * best_scale)
        x, y = best_loc
        print(f"  [{os.path.basename(template_path)}] 발견 (신뢰도: {best_val:.2f}, 배율: {best_scale:.2f})")
        return (x, y, nw, nh, best_scale)

    print(f"  [{os.path.basename(template_path)}] 미발견 (최고 신뢰도: {best_val:.2f})")
    return None


# ── 초대 코드 숫자 인식 ──────────────────────────────────

def find_and_ocr_number() -> str:
    """
    화면에서 번호.png(초대 코드 박스)를 찾고,
    숫자가 있는 하단 영역만 OCR해서 반환.
    """
    screenshot = ImageGrab.grab()
    screen_np  = np.array(screenshot)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_RGB2GRAY)

    found = find_on_screen(TEMPLATE_HEADER, screen_gray)
    if not found:
        print("  [디버그] 번호.png 템플릿을 화면에서 찾지 못함")
        return ""

    x, y, w, h, scale = found

    # header 아래 = 숫자 영역
    # header 높이(h) / HEADER_RATIO = 전체 박스 높이
    total_h = int(h / HEADER_RATIO)
    num_y = y + h
    num_h = total_h - h
    num_region = screenshot.crop((x, num_y, x + w, num_y + num_h))

    # 디버그: 저장
    os.makedirs(DEBUG_DIR, exist_ok=True)
    num_region.save(os.path.join(DEBUG_DIR, "raw.png"))
    print(f"  [디버그] 캡처 저장: {DEBUG_DIR}/raw.png  (좌표: x={x} y={num_y} w={w} h={num_h})")

    import io
    buf = io.BytesIO()
    num_region.save(buf, format="PNG")
    text = ocr.classification(buf.getvalue())
    digits = "".join(filter(str.isdigit, text))
    print(f"  [디버그] OCR 결과: '{text}' → 숫자: '{digits}'")
    return digits


# ── 메인 실행 ────────────────────────────────────────────

_running = False

def run_auto_loop():
    global _running
    _running = True
    print("[F9] 자동 대기 시작... (F9 다시 누르면 중지)\n")

    while _running:
        number = find_and_ocr_number()
        if not number:
            time.sleep(1)
            continue

        print(f"[인식된 숫자] {number}")
        pyperclip.copy(number)

        screenshot = ImageGrab.grab()
        screen_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

        inp = find_on_screen(TEMPLATE_INPUT, screen_gray)
        if not inp:
            time.sleep(1)
            continue

        cx, cy = inp[0] + inp[2] // 2, inp[1] + inp[3] // 2
        pyautogui.click(cx, cy)
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(0.3)

        screen_gray2 = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2GRAY)
        jn = find_on_screen(TEMPLATE_JOIN, screen_gray2)
        if jn:
            pyautogui.click(jn[0] + jn[2] // 2, jn[1] + jn[3] // 2)

        print(f"[완료] '{number}' 입력 완료!\n")
        time.sleep(1)

    print("[중지] 자동 대기 종료\n")


def test_number_ocr():
    """F2: 초대 코드 숫자 인식 테스트 + 클립보드 저장"""
    print("[F2 테스트] 초대 코드 숫자 인식 중...")
    number = find_and_ocr_number()
    if number:
        pyperclip.copy(number)
        print(f"[성공] 인식된 숫자: {number}  ← 클립보드에 복사됨\n")
    else:
        print("[실패] 숫자를 인식하지 못했습니다.\n")


def test_input_field():
    """F3: 입력창 탐색 테스트"""
    print("[F3 테스트] 입력창 탐색 중...")
    screen_gray = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2GRAY)
    found = find_on_screen(TEMPLATE_INPUT, screen_gray)
    if found:
        cx, cy = found[0] + found[2] // 2, found[1] + found[3] // 2
        print(f"[성공] 입력창 발견 위치: ({cx}, {cy})\n")
    else:
        print("[실패] 입력창을 찾지 못했습니다.\n")


def test_join_button():
    """F4: 참가 버튼 탐색 테스트"""
    print("[F4 테스트] 참가 버튼 탐색 중...")
    screen_gray = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2GRAY)
    found = find_on_screen(TEMPLATE_JOIN, screen_gray)
    if found:
        cx, cy = found[0] + found[2] // 2, found[1] + found[3] // 2
        print(f"[성공] 참가 버튼 발견 위치: ({cx}, {cy})\n")
    else:
        print("[실패] 참가 버튼을 찾지 못했습니다.\n")


def on_f2():
    threading.Thread(target=test_number_ocr, daemon=True).start()

def on_f3():
    threading.Thread(target=test_input_field, daemon=True).start()

def on_f4():
    threading.Thread(target=test_join_button, daemon=True).start()

def on_f9():
    global _running
    if _running:
        _running = False
    else:
        threading.Thread(target=run_auto_loop, daemon=True).start()


def main():
    print("=" * 45)
    print("  유튜브 시참 번호 자동 입력 프로그램")
    print("=" * 45)
    print("  F2  - [테스트] 숫자 인식 + 클립보드 저장")
    print("  F3  - [테스트] 입력창 탐색")
    print("  F4  - [테스트] 참가 버튼 탐색")
    print("  F9  - 자동 대기 시작/중지")
    print("  ESC - 종료")
    print("=" * 45 + "\n")

    keyboard.add_hotkey("F2", on_f2)
    keyboard.add_hotkey("F3", on_f3)
    keyboard.add_hotkey("F4", on_f4)
    keyboard.add_hotkey("F9", on_f9)
    keyboard.wait("esc")
    print("종료합니다.")


if __name__ == "__main__":
    main()
