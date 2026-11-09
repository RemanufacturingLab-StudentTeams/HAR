@echo off
rem Alias voor run_compare_models.bat: taakbewust benchmarkrapport.
rem Gebruik bijvoorbeeld: run_benchmark.bat --task action_classification
call "%~dp0run_compare_models.bat" %*
exit /b %ERRORLEVEL%
