@echo off
setlocal

:: This script runs all python scripts in the src\training directory twice,
:: providing different inputs for each run.

:: Define the directory where the Python scripts are located.
set "SCRIPT_DIR=src\training"

:: Check if the directory exists.
if not exist "%SCRIPT_DIR%" (
    echo.
    echo ERROR: The script directory was not found!
    echo       Please ensure this batch file is in your project's root directory.
    echo       Expected path: %SCRIPT_DIR%
    echo.
    pause
    exit /b 1
)

echo Starting the training process for all scripts in %SCRIPT_DIR%...
echo.

:: Loop through each python file in the specified directory.
FOR %%F IN ("%SCRIPT_DIR%\*.py") DO (
    echo =================================================================
    echo Processing script: %%~nxF
    echo =================================================================
    echo.

    :: --- RUN 1: CSTR ---
    echo [RUN 1/2] Executing with Process Unit: 'cstr' and Folds: 10
    (
        echo cstr
        echo 10
    ) | python "%%F"
    echo.
    echo [RUN 1/2] Completed for %%~nxF.
    echo.

    :: --- RUN 2: CLARIFIER ---
    echo [RUN 2/2] Executing with Process Unit: 'clarifier' and Folds: 10
    (
        echo clarifier
        echo 10
    ) | python "%%F"
    echo.
    echo [RUN 2/2] Completed for %%~nxF.
    echo.
)

echo =================================================================
echo All scripts have been processed.
echo =================================================================
echo.
pause