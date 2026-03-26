@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%filt2.py"
set "VENV_PYTHON=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if not exist "%SCRIPT_PATH%" (
    echo ERROR: Could not find filt2.py at "%SCRIPT_PATH%"
    exit /b 1
)

if "%~1"=="" (
    set "TARGET_DIR=%CD%"
) else (
    set "TARGET_DIR=%~1"
)

if not exist "%TARGET_DIR%" (
    echo ERROR: Directory does not exist: "%TARGET_DIR%"
    exit /b 1
)

if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
) else (
    set "PYTHON_CMD=python"
)

set /a PROCESSED=0

echo Using Python: "%PYTHON_CMD%"
echo Target directory: "%TARGET_DIR%"
echo.

for %%F in ("%TARGET_DIR%\*.csv") do (
    set "FILE_PATH=%%~fF"
    set "FILE_NAME=%%~nF"

    if /I "!FILE_NAME:~-9!"=="_original" (
        echo Skipping original file: "%%~nxF"
    ) else if /I "!FILE_NAME:~-15!"=="_repair_details" (
        echo Skipping details file: "%%~nxF"
    ) else (
        set "ORIGINAL_PATH=%%~dpF%%~nF_original%%~xF"
        if exist "!ORIGINAL_PATH!" (
            set "INPUT_PATH=!ORIGINAL_PATH!"
            echo Processing with existing original: "!INPUT_PATH!"
        ) else (
            set "INPUT_PATH=!FILE_PATH!"
            echo Processing: "!INPUT_PATH!"
        )

        "%PYTHON_CMD%" "%SCRIPT_PATH%" "!INPUT_PATH!"
        if errorlevel 1 (
            echo ERROR: Processing failed for "!INPUT_PATH!"
        ) else (
            set /a PROCESSED+=1
        )
        echo.
    )
)

echo Completed. Successfully processed !PROCESSED! file(s).
endlocal
