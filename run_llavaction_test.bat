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
rem LLaVAction 0.0.1 is tegen transformers 4.x geschreven; de adapter
rem voorziet compatibility-shims. Controleer daarom alleen of het pakket
rem geinstalleerd is, zonder het hier al te importeren.
"%PYTHON%" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('llavaction') else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: llavaction ontbreekt; draai met --upgrade om
    echo requirements.txt opnieuw te installeren.
    popd
    exit /b 1
)
if not exist "models" mkdir "models"
if not exist "third_party" mkdir "third_party"
if not exist "third_party\llavaction_runtime\transformers\__init__.py" (
    echo [3/5] LLaVAction-runtime ^(transformers 4.x^) installeren...
    "%PYTHON%" -m pip install --quiet --disable-pip-version-check --no-deps --target third_party\llavaction_runtime -r requirements-llavaction.txt
    if errorlevel 1 (
        echo ERROR: installeren van de LLaVAction-runtime is mislukt.
        popd
        exit /b 1
    )
)
echo LLaVAction-7B is een 7B-model ^(~15 GB download^) en praktisch GPU-only.
echo Het model valt onder de CC-BY-NC-SA-4.0-licentie ^(niet-commercieel^).
echo [4/4] LLaVAction-actieclassificatie uitvoeren...
"%PYTHON%" -u src\run_action_test.py --config configs\baseline.yaml --model llavaction %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Actievoorspellingen staan in outputs\Disassembly1\llavaction
) else (
    echo.
    echo ERROR: De LLaVAction-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
