@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, clip" 2>nul
if errorlevel 1 (
    echo ERROR: torch/clip ontbreken; draai met --upgrade om requirements.txt
    echo opnieuw te installeren.
    popd
    exit /b 1
)
echo ActionCLIP downloadt bij de eerste run de OpenAI-CLIP ViT-gewichten en
echo het ActionCLIP Kinetics-checkpoint ^(~430 MB^) naar models\.
echo [4/4] ActionCLIP-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model actionclip %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\actionclip
) else (
    echo.
    echo ERROR: De ActionCLIP-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
