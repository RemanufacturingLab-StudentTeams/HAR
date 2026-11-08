@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import ultralytics; from ultralytics.models.sam import SAM3SemanticPredictor" 2>nul
if errorlevel 1 (
    echo ERROR: ultralytics met SAM3-ondersteuning ontbreekt; draai met --upgrade.
    popd
    exit /b 1
)
if exist "models\sam3.1_multiplex.pt" (
    echo Lokaal SAM 3.1-checkpoint gevonden: models\sam3.1_multiplex.pt ^(geen Hugging Face-login nodig^).
) else (
    echo Let op: models\sam3.1_multiplex.pt ontbreekt; de SAM 3.1-gewichten van facebook/sam3.1 ^(sam3.1_multiplex.pt^) vereisen een goedgekeurde
    echo Hugging Face-toegangsaanvraag en een login via 'huggingface-cli login' ^(of 'hf auth login'^).
)
echo [4/4] SAM 3.1-test uitvoeren...
"%PYTHON%" -u src\run_sam31_video.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Resultaten staan in outputs\Disassembly1\sam31
) else (
    echo.
    echo ERROR: De SAM 3.1-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
