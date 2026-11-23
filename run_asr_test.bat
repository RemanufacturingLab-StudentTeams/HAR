@echo off
rem Spraakannotatie: draait Vosk en/of Whisper op een eigen opname.
rem Gebruikt een eigen omgeving (.venv-asr), zodat de hoofd-.venv (torch/transformers) onaangeroerd blijft.
rem Voorbeelden:
rem   run_asr_test.bat data\opname_3min.m4a --combi small --vanaf 88
rem   run_asr_test.bat data\opname_3min.m4a --models vosk,small,medium
setlocal
pushd "%~dp0"
if not exist "models" mkdir "models"
if not exist "data" mkdir "data"
if not exist "outputs" mkdir "outputs"
set "PYTHON=%CD%\.venv-asr\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [1/3] Omgeving .venv-asr maken...
    py -m venv .venv-asr 2>nul || python -m venv .venv-asr
    if not exist "%PYTHON%" (
        echo ERROR: kon .venv-asr niet maken. Is Python geinstalleerd?
        popd & exit /b 1
    )
    echo [2/3] Pakketten installeren uit requirements-asr.txt...
    "%PYTHON%" -m pip install --disable-pip-version-check -r requirements-asr.txt
    if errorlevel 1 ( popd & exit /b 1 )
)
if not exist "models\vosk-model-small-nl-0.22" (
    echo [3/3] Vosk-model downloaden...
    "%PYTHON%" src\download_vosk.py
)
if "%~1"=="" (
    echo Gebruik: run_asr_test.bat data\opname.m4a [opties]
    echo   run_asr_test.bat data\opname_3min.m4a --combi small --vanaf 88
    echo   run_asr_test.bat data\opname_3min.m4a --models vosk,small,medium
    popd & exit /b 1
)
"%PYTHON%" src\run_asr_test.py %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
