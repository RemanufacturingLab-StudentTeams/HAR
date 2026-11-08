@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import sys; sys.path.insert(0, r'third_party\VITRA'); import vitra.models" 2>nul
if errorlevel 1 (
    echo ERROR: de VITRA-repo is niet gevonden of niet importeerbaar.
    echo Clone hem eenmalig vanuit de projectmap:
    echo   git clone https://github.com/microsoft/VITRA.git third_party\VITRA
    echo Installeer daarna de runtime-dependencies met:
    echo   run_vitra_test.bat --upgrade
    popd
    exit /b 1
)
"%PYTHON%" -c "import sentencepiece, accelerate, transformers, scipy, smplx, pytorch_lightning, omegaconf, skimage, utils3d, yacs, trimesh" 2>nul
if errorlevel 1 (
    echo ERROR: VITRA-runtime ^(sentencepiece/accelerate/transformers/scipy/HaWoR-dependencies^) ontbreekt;
    echo draai dit script met --upgrade om requirements.txt opnieuw te installeren.
    popd
    exit /b 1
)
echo Let op: VITRA laadt Paligemma2 ^(google/paligemma2-3b-mix-224^) en heeft daarvoor
echo Hugging Face-toegang nodig. Vraag die aan op de modelpagina en log in met 'hf auth login'.
echo Het 3B-checkpoint ^(~12 GB^) wordt bij de eerste run naar de HF-cache gedownload.
echo Let op: de HaWoR/MANO-handreconstructie vereist MANO_RIGHT.pkl in
echo third_party\VITRA\weights\mano ^(registratie/licentie via https://mano.is.tue.mpg.de/^).
echo Zonder dat bestand stopt pass 1 met een uitleg; de VITRA-inferentie zelf kan wel verder
echo met een bestaand statebestand ^(vitra_state_path of vitra_states_path^).
echo Let op: VITRA is getraind met transformers 4.47.1. Deze stap draait daarom in
echo een subprocess met third_party\vitra_runtime op de PYTHONPATH; de hoofd-venv
echo blijft op transformers 5.x voor SAM3.
if not exist "third_party\vitra_runtime\transformers" (
    echo [3/5] VITRA-runtime ^(transformers 4.47.1^) installeren...
    "%PYTHON%" -m pip install --quiet --disable-pip-version-check --no-deps --target third_party\vitra_runtime -r requirements-vitra.txt
    if errorlevel 1 (
        echo ERROR: installeren van de VITRA-runtime is mislukt.
        popd
        exit /b 1
    )
)
echo [4/4] VITRA-test uitvoeren...
"%PYTHON%" -u src\run_vitra_test.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\vitra
) else (
    echo.
    echo ERROR: De VITRA-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
