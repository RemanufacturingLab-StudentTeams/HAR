@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, transformers" 2>nul
if errorlevel 1 (
    echo ERROR: torch/transformers ontbreken; draai met --upgrade om
    echo requirements.txt opnieuw te installeren.
    popd
    exit /b 1
)
echo X-CLIP downloadt bij de eerste run microsoft/xclip-base-patch16-zero-shot
echo naar de Hugging Face-cache ^(eenmalig, enkele honderden MB^).
echo [4/4] X-CLIP-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model xclip %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\xclip
) else (
    echo.
    echo ERROR: De X-CLIP-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
