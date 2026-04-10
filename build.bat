@echo off
echo EXE 빌드 중...

pyinstaller ^
  --onefile ^
  --name live-code-catcher ^
  --add-data "images;images" ^
  --collect-all ddddocr ^
  --collect-all onnxruntime ^
  --hidden-import mss ^
  --hidden-import pynput ^
  auto_number.py

echo.
echo 완료! dist\live-code-catcher.exe 확인하세요.
pause
