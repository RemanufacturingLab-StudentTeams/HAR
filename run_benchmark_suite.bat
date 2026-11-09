@echo off
rem Automatisering: eerst de unittests, daarna het taakbewuste benchmarkrapport.
rem Gebruik: run_benchmark_suite.bat [--task action_classification] [--upgrade]
setlocal
call "%~dp0run_unit_tests.bat" %*
if errorlevel 1 (
    echo ERROR: Unittests zijn mislukt; benchmarkrapport wordt niet gedraaid.
    exit /b 1
)
call "%~dp0run_compare_models.bat" %*
exit /b %ERRORLEVEL%
