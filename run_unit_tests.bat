@echo off
rem Draai de automatische unittests (zonder model/GPU) met een dubbelklik.
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
echo [4/4] Unittests uitvoeren...
"%PYTHON%" -m unittest discover -s tests -v
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. Alle unittests zijn geslaagd.
) else (
    echo.
    echo ERROR: Een of meer unittests zijn mislukt ^(code %EXIT_CODE%^).
)
popd
exit /b %EXIT_CODE%
