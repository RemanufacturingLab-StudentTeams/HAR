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
    echo ERROR: torch/transformers ontbreken; draai met --upgrade.
    popd
    exit /b 1
)
echo Let op: Qwen3-VL is een algemene VLM, geen gespecialiseerde temporele
echo actieclassifier. Het label en de start/end-tijden worden apart gemeten.
echo Qwen/Qwen3-VL-8B-Instruct wordt bij de eerste run gedownload ^(eenmalig^).
echo [4/4] Qwen3-VL-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model qwen3vl %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\qwen3vl
) else (
    echo.
    echo ERROR: De Qwen3-VL-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
