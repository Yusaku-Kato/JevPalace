@echo off
rem JevPalace Decision API (http://127.0.0.1:8700)
rem   start-judge.bat              ... config.yaml の設定で起動 (既定: テキスト専用 = 高速)
rem   start-judge.bat --vision     ... 画像対応モードで起動
rem   start-judge.bat --backend X  ... config.yaml の別バックエンド (モデル) で起動
cd /d "%~dp0"
set PIP_CACHE_DIR=%~dp0.cache\pip
set PYTHONIOENCODING=utf-8
if not exist judge\.venv\Scripts\python.exe (
  echo [setup] creating judge\.venv ...
  python -m venv judge\.venv || exit /b 1
  judge\.venv\Scripts\python.exe -m pip install -r judge\requirements.txt || exit /b 1
)
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8700"
cd judge
.venv\Scripts\python.exe -m app.main %*
