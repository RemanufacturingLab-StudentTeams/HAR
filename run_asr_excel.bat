@echo off
rem Zet de resultaten van run_asr_test.bat om naar een Excel-bestand voor de spreker.
rem Voorbeeld: run_asr_excel.bat outputs\opname_3min --vanaf 88
setlocal
pushd "%~dp0"
set "PYTHON=%CD%\.venv-asr\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: .venv-asr ontbreekt. Draai eerst run_asr_test.bat.
    popd & exit /b 1
)
if "%~1"=="" (
    echo Gebruik: run_asr_excel.bat outputs\opname_3min [--vanaf 88]
    popd & exit /b 1
)
"%PYTHON%" src\asr_naar_excel.py %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
