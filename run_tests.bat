@ECHO OFF
REM https://stackoverflow.com/questions/886848/how-to-make-windows-batch-file-pause-when-double-clicked/12036163

if "%parent%"=="" set parent=%~0
if "%console_mode%"=="" (set console_mode=1& for %%x in (%cmdcmdline%) do if /i "%%~x"=="/c" set console_mode=0)

pytest --exitfirst --verbose --failed-first --cov=. --cov-report html

if "%parent%"=="%~0" ( if "%console_mode%"=="0" pause )
