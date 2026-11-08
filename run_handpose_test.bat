@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import mediapipe" 2>nul
if errorlevel 1 (
    echo ERROR: mediapipe ontbreekt; draai met --upgrade om requirements opnieuw te installeren.
    popd
    exit /b 1
)
echo [4/4] MediaPipe handpose-test uitvoeren...
"%PYTHON%" -u src\run_handpose.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Handlandmarks staan in outputs\Disassembly1
) else (
    echo.
    echo ERROR: De MediaPipe-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
