@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, torchvision, transformers" 2>nul
if errorlevel 1 (
    echo ERROR: InternVideo3-dependencies ontbreken ^(torch/torchvision/
    echo transformers^); draai met --upgrade.
    popd
    exit /b 1
)
echo InternVideo3 downloadt bij de eerste run yanziang/InternVideo3-8B-Instruct
echo naar de Hugging Face-cache ^(remote code, eenmalig, ~18 GB in bf16^).
echo [4/4] InternVideo3-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model internvideo3 %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\internvideo3
) else (
    echo.
    echo ERROR: De InternVideo3-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
