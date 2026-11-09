@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, transformers, timm, einops, open_clip" 2>nul
if errorlevel 1 (
    echo ERROR: InternVideo2-dependencies ontbreken ^(torch/transformers/
    echo timm/einops/open_clip_torch^); draai met --upgrade.
    popd
    exit /b 1
)
echo InternVideo2 downloadt bij de eerste run OpenGVLab/InternVideo2_CLIP_S
echo naar de Hugging Face-cache ^(remote code, eenmalig^).
echo flash-attn is niet nodig: de configuratie gebruikt gewone attention.
echo [4/4] InternVideo2-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model internvideo2 %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\internvideo2
) else (
    echo.
    echo ERROR: De InternVideo2-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
