@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, transformers, num2words" 2>nul
if errorlevel 1 (
    echo ERROR: SmolVLM2-dependencies ontbreken ^(torch/transformers/
    echo num2words^); draai met --upgrade.
    popd
    exit /b 1
)
echo SmolVLM2 downloadt bij de eerste run HuggingFaceTB/SmolVLM2-2.2B-Instruct
echo naar de Hugging Face-cache ^(eenmalig, ~5 GB in bf16^).
echo [4/4] SmolVLM2-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model smolvlm2 %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\smolvlm2
) else (
    echo.
    echo ERROR: De SmolVLM2-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
