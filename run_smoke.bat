@echo off
REM One-scene smoke test of every archive. Run after setup_windows.bat and after the credential files exist.
cd /d %~dp0\..
call venv\Scripts\activate
echo [1/2] Dry run: fetch IBTrACS and print the storm centre (no satellite downloads)
python -m milton_da.scripts.prepare_milton --root data\milton --times 2024-10-07T18:00 --dry-run
if errorlevel 1 goto fail
echo.
echo [2/2] Real run for one analysis time: GOES-16, ATMS, IMERG, ERA5, then build the scene
python -m milton_da.scripts.prepare_milton --root data\milton --times 2024-10-07T18:00
if errorlevel 1 goto fail
echo.
echo Success. Scene written under data\milton\scenes. Now run the full case:
echo   python -m milton_da.scripts.prepare_milton --root data\milton --times 2024-10-06T12 2024-10-06T18 2024-10-07T00 2024-10-07T06 2024-10-07T12 2024-10-07T18 2024-10-07T20 2024-10-08T00 2024-10-08T06 2024-10-08T12 2024-10-08T18 2024-10-09T00 2024-10-09T06 2024-10-09T12 2024-10-09T18 2024-10-10T00
pause
exit /b 0
:fail
echo.
echo Something failed. Copy the traceback above and paste it into the Claude conversation.
pause
exit /b 1
