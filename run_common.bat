@echo off
rem Gedeelde setup voor alle run_*_test.bat scripts: venv + pip fast-check.
rem Gebruik: call run_common.bat [--upgrade] [--upgrade=...]
rem Na afloop zijn VENV_DIR en PYTHON gezet en is de projectmap actief (pushd).
rem Retourneert ERRORLEVEL 1 bij een fout; de caller moet dan stoppen.
rem
rem Portability: werkt met de py-launcher, maar ook met python/python3 op
rem PATH. Een .venv van een andere pc of Python-versie wordt automatisch
rem opnieuw aangemaakt. Git is alleen nodig als packages geinstalleerd
rem moeten worden (requirements bevat git-URL's).
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Kan de projectmap niet openen.
    exit /b 1
)

rem Mappen die niet in git staan maar wel nodig zijn voor modeldownloads.
if not exist "models" mkdir "models"
if not exist "data" mkdir "data"
if not exist "outputs" mkdir "outputs"

set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "HASH_FILE=%VENV_DIR%\.duplo_req_hash.txt"
set "FORCE_UPGRADE=0"
:scan_args
if "%~1"=="" goto args_scanned
set "CUR=%~1"
if "%CUR%"=="--upgrade" set "FORCE_UPGRADE=1"
if "%CUR:~0,10%"=="--upgrade=" set "FORCE_UPGRADE=1"
shift
goto scan_args
:args_scanned

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

rem --- Python zoeken: py-launcher eerst, daarna python/python3 op PATH ---
set "PYTHON_CMD="
call :find_python py -3
if not defined PYTHON_CMD call :find_python python
if not defined PYTHON_CMD call :find_python python3
if not defined PYTHON_CMD (
    echo ERROR: Python 3.10 of nieuwer is niet gevonden.
    echo Installeer Python via https://www.python.org/downloads/windows/
    echo en vink tijdens de installatie "Add python.exe to PATH" aan.
    exit /b 1
)

rem --- Bestaande venv controleren; een venv van een andere pc is onbruikbaar ---
if exist "%VENV_DIR%" (
    if not exist "%PYTHON%" (
        echo [1/3] Bestaande .venv hoort niet bij deze pc; opnieuw aanmaken...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        "%PYTHON%" -c "import sys" >nul 2>&1
        if errorlevel 1 (
            echo [1/3] Bestaande .venv is beschadigd; opnieuw aanmaken...
            rmdir /s /q "%VENV_DIR%"
        )
    )
)

set "CREATED_VENV=0"
if not exist "%PYTHON%" (
    echo [1/3] Virtuele omgeving aanmaken met !PYTHON_CMD!...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Aanmaken van de virtuele omgeving is mislukt.
        echo Controleer of Python 3.10+ correct is geinstalleerd.
        exit /b 1
    )
    set "CREATED_VENV=1"
) else (
    echo [1/3] Bestaande virtuele omgeving gebruiken.
)

rem --- pip fast-check: sla install over als requirements ongewijzigd zijn ---
rem De hash wordt met de venv-Python berekend (geen certutil nodig), zodat
rem dit ook werkt op systemen waar certutil ontbreekt of lokaal is vertaald.
set "SKIP_INSTALL=0"
set "CURRENT_HASH="
set "SAVED_HASH="
if "%FORCE_UPGRADE%"=="0" if exist "%HASH_FILE%" if exist "requirements.txt" (
    "%PYTHON%" -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())" > "%VENV_DIR%\.duplo_req_hash.tmp" 2>nul
    if exist "%VENV_DIR%\.duplo_req_hash.tmp" (
        set /p CURRENT_HASH=<"%VENV_DIR%\.duplo_req_hash.tmp"
        del "%VENV_DIR%\.duplo_req_hash.tmp" >nul 2>&1
    )
    set /p SAVED_HASH=<"%HASH_FILE%"
    if defined CURRENT_HASH if "!CURRENT_HASH!"=="!SAVED_HASH!" set "SKIP_INSTALL=1"
)

if "%SKIP_INSTALL%"=="1" (
    echo [2/3] Requirements ongewijzigd, pip install overgeslagen.
) else (
    echo [2/3] Python packages controleren en installeren...
    where git >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Git is nodig voor een aantal requirements ^(git-URL's^).
        echo Installeer Git via https://git-scm.com/download/win en probeer opnieuw.
        exit /b 1
    )
    if "%CREATED_VENV%"=="1" (
        "%PYTHON%" -m pip install --quiet --prefer-binary --disable-pip-version-check --upgrade pip
    )
    "%PYTHON%" -m pip install --prefer-binary --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Een of meer packages konden niet worden geinstalleerd.
        echo Controleer je internetverbinding en of Python 3.10+ geschikt is
        echo voor alle packages; draai dit script daarna opnieuw.
        exit /b 1
    )
    "%PYTHON%" -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())" > "%HASH_FILE%" 2>nul
)

echo [3/3] Imports verifieren...
"%PYTHON%" -c "import cv2, numpy, pandas, torch, yaml, tqdm; print('Alle vereiste imports zijn beschikbaar.')"
if errorlevel 1 (
    echo ERROR: De Python-omgeving is niet compleet.
    exit /b 1
)

rem Exporteer variabelen naar de caller en behoud pushd (endlocal met set).
endlocal & set "VENV_DIR=%VENV_DIR%" & set "PYTHON=%PYTHON%" & set "PYTHONUTF8=1" & set "PYTHONIOENCODING=utf-8" & set "DUPLO_COMMON_OK=1"
rem Let op: pushd van hierboven blijft actief; de caller doet popd.
exit /b 0

:find_python
where %~1 >nul 2>&1
if errorlevel 1 exit /b 1
%~1 %~2 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Waarschuwing: %~1 %~2 is ouder dan Python 3.10 en wordt overgeslagen.
    exit /b 1
)
set "PYTHON_CMD=%~1 %~2"
exit /b 0
