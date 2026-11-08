@echo off
rem Gedeelde setup voor alle run_*_test.bat scripts: venv + pip fast-check.
rem Gebruik: call run_common.bat [--upgrade] [--upgrade=...]
rem Na afloop zijn VENV_DIR en PYTHON gezet en is de projectmap actief (pushd).
rem Retourneert ERRORLEVEL 1 bij een fout; de caller moet dan stoppen.
setlocal EnableDelayedExpansion

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Kan de projectmap niet openen.
    exit /b 1
)

set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "FORCE_UPGRADE=0"
:scan_args
if "%~1"=="" goto args_scanned
set "CUR=%~1"
if "%CUR%"=="--upgrade" set "FORCE_UPGRADE=1"
if "%CUR:~0,10%"=="--upgrade=" set "FORCE_UPGRADE=1"
shift
goto scan_args
:args_scanned

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher ^(py^) is niet gevonden.
    echo Installeer Python 3.10 of nieuwer en probeer opnieuw.
    exit /b 1
)

set "CREATED_VENV=0"
if not exist "%PYTHON%" (
    echo [1/3] Virtuele omgeving aanmaken...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Aanmaken van de virtuele omgeving is mislukt.
        exit /b 1
    )
    set "CREATED_VENV=1"
) else (
    echo [1/3] Bestaande virtuele omgeving gebruiken.
)

rem --- pip fast-check: sla install over als requirements ongewijzigd zijn ---
rem Alleen hex-regels tellen als hash (findstr-guard), zodat certutil-fouttekst
rem nooit als hash wordt vergeleken; bij een lege hash volgt altijd een
rem (veilige) pip-install in plaats van een foute skip.
set "HASH_FILE=%VENV_DIR%\.duplo_req_hash.txt"
set "SKIP_INSTALL=0"
set "CURRENT_HASH="
if "%FORCE_UPGRADE%"=="0" if exist "%HASH_FILE%" if exist "requirements.txt" (
    for /f "delims=" %%H in ('certutil -hashfile "requirements.txt" SHA256 ^| findstr /r /c:"^[0-9a-f ][0-9a-f ]*$"') do set "CURRENT_HASH=%%H"
    if defined CURRENT_HASH (
        set /p SAVED_HASH=<"%HASH_FILE%"
        if "!CURRENT_HASH!"=="!SAVED_HASH!" set "SKIP_INSTALL=1"
    )
)

if "%SKIP_INSTALL%"=="1" (
    echo [2/3] Requirements ongewijzigd, pip install overgeslagen.
) else (
    echo [2/3] Python packages controleren en installeren...
    if "%CREATED_VENV%"=="1" (
        "%PYTHON%" -m pip install --quiet --disable-pip-version-check --upgrade pip
    )
    "%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Een of meer packages konden niet worden geinstalleerd.
        exit /b 1
    )
    set "NEW_HASH="
    for /f "delims=" %%H in ('certutil -hashfile "requirements.txt" SHA256 ^| findstr /r /c:"^[0-9a-f ][0-9a-f ]*$"') do set "NEW_HASH=%%H"
    if defined NEW_HASH (
        (echo !NEW_HASH!)>"%HASH_FILE%"
    ) else (
        echo Waarschuwing: requirements-hash kon niet worden berekend; volgende run controleert opnieuw.
        del "%HASH_FILE%" 2>nul
    )
)

echo [3/3] Imports verifieren...
"%PYTHON%" -c "import cv2, pandas, yaml, tqdm; print('Alle vereiste imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

rem Exporteer variabelen naar de caller en behoud pushd (endlocal met set).
endlocal & set "VENV_DIR=%VENV_DIR%" & set "PYTHON=%PYTHON%" & set "DUPLO_COMMON_OK=1"
rem Let op: pushd van hierboven blijft actief; de caller doet popd.
exit /b 0
