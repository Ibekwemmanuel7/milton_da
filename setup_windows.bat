@echo off
REM One-time setup for milton_da on Windows. Run from C:\Users\taylo (the folder that CONTAINS milton_da).
cd /d %~dp0\..
echo Working directory: %CD%
if not exist venv (
    py -3.11 -m venv venv || py -3 -m venv venv
)
call venv\Scripts\activate
python -m pip install --upgrade pip
REM CPU build of torch; replace the index-url with https://download.pytorch.org/whl/cu121 if you have an NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r milton_da\requirements.txt
echo.
echo Running the offline test suite (about 30 s)...
python -m pytest -q milton_da\tests
echo.
echo Setup done. Next: edit %USERPROFILE%\_netrc and %USERPROFILE%\.cdsapirc (see milton_da\WINDOWS_SETUP.md), then run milton_da\run_smoke.bat
pause
