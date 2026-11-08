@echo off
rem Wrapper om run_common.bat: alle argumenten (%*) gaan door naar Python,
rem inclusief --upgrade (bat-level flag; Python negeert hem via cli.py).
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
echo [4/4] Modelvergelijking uitvoeren...
"%PYTHON%" -u src\compare_models.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Het rapport staat in outputs\Disassembly1\comparison
) else (
    echo.
    echo ERROR: De modelvergelijking is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
