"""
운빨존많겜 시참 코드 자동 입력 프로그램

F2  - 초대 코드 위치 등록 (숫자 위에 마우스 올리고 누르기)
F3  - 입력창 위치 등록 (입력창 위에 마우스 올리고 누르기)
F4  - 참가 버튼 위치 등록 (버튼 위에 마우스 올리고 누르기)
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

DEBUG_DIR   = "debug"
CONFIG_FILE = "config.json"

# 등록된 위치
capture_region = None  # (x, y, w, h) - 초대 코드 OCR 영역
input_pos      = None  # (cx, cy)     - 입력창 클릭 위치
join_pos       = None  # (cx, cy)     - 참가 버튼 클릭 위치


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
    global capture_region, input_pos, join_pos
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        capture_region = data.get("capture_region")
        input_pos      = data.get("input_pos")
        join_pos       = data.get("join_pos")
        if capture_region:
            print(f"[설정 로드] 코드: {capture_region}")
        if input_pos:
            print(f"[설정 로드] 입력창: {input_pos}")
        if join_pos:
            print(f"[설정 로드] 참가 버튼: {join_pos}")


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump({
            "capture_region": capture_region,
            "input_pos": input_pos,
            "join_pos": join_pos,
        }, f, indent=2)


# ── 초대 코드 OCR ────────────────────────────────────────

def ocr_number() -> str:
    if not capture_region:
        print("[오류] F2로 초대 코드 위치를 먼저 설정하세요.")
        return ""

    img = grab_pil(region=capture_region)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    img.save(os.path.join(DEBUG_DIR, "raw.png"))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    text = ocr.classification(buf.getvalue())
    text = text.replace('o', '0').replace('O', '0') \
               .replace('l', '1').replace('I', '1') \
               .replace('s', '5').replace('S', '5') \
               .replace('g', '9').replace('q', '9') \
               .replace('b', '6').replace('B', '8')
    digits = "".join(filter(str.isdigit, text))
    print(f"  [OCR] '{text}' → '{digits}'")
    return digits


# ── F2: 초대 코드 위치 등록 ───────────────────────────────

def save_capture_pos():
    global capture_region
    cx, cy = pyautogui.position()

    scan = 200
    arr = grab_screen(region=(cx - scan, cy - scan, scan * 2, scan * 2))
    hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 120]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (x <= scan <= x + w and y <= scan <= y + h):
            continue
        if w < h * 0.8:
            continue
        score = w * h
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best:
        bx, by, bw, bh = best
        pad = 4
        capture_region = (
            (cx - scan) + bx - pad,
            (cy - scan) + by - pad,
            bw + pad * 2,
            bh + pad * 2,
        )
        print(f"[F2] 박스 감지: {bw}x{bh} → {capture_region[2]}x{capture_region[3]} at ({capture_region[0]},{capture_region[1]})")
    else:
        capture_region = (cx - 55, cy - 25, 110, 50)
        print(f"[F2] 감지 실패 → 기본값 사용: {capture_region}")

    save_config()
    print()


# ── F3: 입력창 위치 등록 ──────────────────────────────────

def save_input_pos():
    global input_pos
    cx, cy = pyautogui.position()
    input_pos = [cx, cy]
    save_config()
    print(f"[F3] 입력창 위치 저장: ({cx}, {cy})\n")


# ── F4: 참가 버튼 위치 등록 ───────────────────────────────

def save_join_pos():
    global join_pos
    cx, cy = pyautogui.position()
    join_pos = [cx, cy]
    save_config()
    print(f"[F4] 참가 버튼 위치 저장: ({cx}, {cy})\n")


# ── 메인 실행 ────────────────────────────────────────────

_running = False


def run_auto_loop():
    global _running
    _running = True
    print("[F9] 자동 대기 시작... (F9 다시 누르면 중지)\n")

    while _running:
        if not capture_region:
            print("[오류] F2로 초대 코드 위치를 먼저 등록하세요.")
            time.sleep(2)
            continue
        if not input_pos:
            print("[오류] F3으로 입력창 위치를 먼저 등록하세요.")
            time.sleep(2)
            continue

        t0 = time.time()
        number = ocr_number()
        t1 = time.time()
        print(f"  OCR: {t1-t0:.2f}s")

        if not number or len(number) != 4:
            time.sleep(1)
            continue

        print(f"[인식된 숫자] {number}")
        pyautogui.click(*input_pos)
        time.sleep(0.05)

        for digit in number:
            keyboard.press_and_release(digit)
        pyautogui.press("enter")
        t2 = time.time()
        print(f"  입력: {t2-t1:.2f}s")

        if join_pos:
            time.sleep(0.05)
            pyautogui.click(*join_pos)

        print(f"[완료] '{number}' 총 {t2-t0:.2f}s / 종료\n")
        os._exit(0)

    print("[중지] 자동 대기 종료\n")


# ── 핫키 ─────────────────────────────────────────────────

def on_f2(): threading.Thread(target=save_capture_pos, daemon=True).start()
def on_f3(): threading.Thread(target=save_input_pos,   daemon=True).start()
def on_f4(): threading.Thread(target=save_join_pos,    daemon=True).start()

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
    print("  F2  - 초대 코드 위치 등록 (숫자 위에 마우스)")
    print("  F3  - 입력창 위치 등록 (입력창 위에 마우스)")
    print("  F4  - 참가 버튼 위치 등록 (버튼 위에 마우스)")
    print("  F9  - 자동 대기 시작 / 중지")
    print("  ESC - 종료")
    print("=" * 45 + "\n")

    load_config()

    keyboard.add_hotkey("F2", on_f2)
    keyboard.add_hotkey("F3", on_f3)
    keyboard.add_hotkey("F4", on_f4)
    keyboard.add_hotkey("F9", on_f9)

    try:
        keyboard.wait("esc")
    except KeyboardInterrupt:
        pass
    print("종료합니다.")


if __name__ == "__main__":
    main()
