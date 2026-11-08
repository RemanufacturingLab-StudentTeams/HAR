@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
"%PYTHON%" -c "import sam3, triton" 2>nul
if errorlevel 1 (
    echo ERROR: sam3 of triton ontbreekt; draai met --upgrade om requirements opnieuw te installeren.
    echo Op Windows is triton-windows nodig; het officiele triton heeft geen Windows-wheels.
    popd
    exit /b 1
)
echo Let op: de SAM3-gewichten van facebook/sam3 vereisen een goedgekeurde
echo Hugging Face-toegangsaanvraag en een login via 'hf auth login'.
echo [4/4] SAM3-test uitvoeren...
"%PYTHON%" -u src\run_sam3_video.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Resultaten staan in outputs\Disassembly1\sam3
) else (
    echo.
    echo ERROR: De SAM3-test is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
