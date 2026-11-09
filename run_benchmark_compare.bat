@echo off
rem Benchmark geselecteerde modellen op één video of één model op meerdere
rem video's. Rapporten worden automatisch gegenereerd.
rem Voorbeelden:
rem   run_benchmark_compare.bat --preset action --video data\Disassembly1.mp4
rem   run_benchmark_compare.bat --preset detection --video data\Disassembly1.mp4
rem   run_benchmark_compare.bat --model xclip --videos data\v1.mp4 data\v2.mp4
setlocal
call "%~dp0run_common.bat" %*
if errorlevel 1 (
    popd
    exit /b 1
)
echo [4/4] Benchmarkvergelijking uitvoeren...
"%PYTHON%" -u src\benchmark_compare.py --config configs\baseline.yaml %*
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo.
    echo Klaar. De rapporten zijn automatisch gegenereerd.
) else (
    echo.
    echo ERROR: De benchmark is gestopt met code %EXIT_CODE%.
)
popd
exit /b %EXIT_CODE%
