@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import torch, torchvision, transformers, timm, einops, sentencepiece, google.protobuf" 2>nul
if errorlevel 1 (
    echo ERROR: InternVideo2.5-dependencies ontbreken ^(torch/torchvision/
    echo transformers/timm/einops/sentencepiece/protobuf^); draai met --upgrade.
    popd
    exit /b 1
)
echo InternVideo2.5 downloadt bij de eerste run OpenGVLab/InternVideo2_5_Chat_8B
echo naar de Hugging Face-cache ^(remote code, eenmalig, ~16 GB in bf16^).
echo flash-attn is optioneel: de adapter gebruikt hem automatisch als de
echo Windows-wheel is geinstalleerd en valt anders terug op eager attention.
if not exist "third_party\vitra_runtime\transformers" (
    echo [3/5] Gepinde transformers 4.47-runtime installeren...
    "%PYTHON%" -m pip install --quiet --disable-pip-version-check --no-deps --target third_party\vitra_runtime -r requirements-vitra.txt
    if errorlevel 1 (
        echo ERROR: installeren van de InternVideo2.5-runtime is mislukt.
        popd
        exit /b 1
    )
)
echo InternVideo2.5 draait in een subprocess met transformers 4.47.1.
echo [4/4] InternVideo2.5-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model internvideo25 %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\internvideo25
) else (
    echo.
    echo ERROR: De InternVideo2.5-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
