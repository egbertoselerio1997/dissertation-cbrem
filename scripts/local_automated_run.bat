@echo off
setlocal

REM ============================================================================
REM  Automated Execution Script for the Complete Pipeline
REM
REM  This script automates the following sequence:
REM  1. Runs Bayesian optimization and final simulation.
REM  2. Trains the surrogate model.
REM  3. Runs the final process optimization.
REM ============================================================================

REM Change directory to the project root to ensure relative paths work.
cd /d "%~dp0.."

REM Define the name for our temporary input file.
set INPUT_FILE=_temp_inputs.txt

echo.
echo ========================================================
echo [1/3] Starting Bayesian Optimization and Simulation
echo ========================================================
echo.

REM --- Create a temporary file with all the required inputs ---
echo Creating temporary input file for the simulation script...
(
    echo y
    echo 1000
    echo 100
    echo 50
    echo 0.99
    echo 10000
) > %INPUT_FILE%

echo Providing automated inputs from %INPUT_FILE%...
echo Running...
echo.

REM --- Run the script, redirecting input FROM the temporary file ---
uv run src\main_simulation\main_with_bayesian_opt.py < %INPUT_FILE%

REM --- Clean up the temporary file ---
echo Deleting temporary input file...
if exist %INPUT_FILE% del %INPUT_FILE%

echo.
echo Bayesian Optimization and Simulation finished.
echo.
echo ========================================================
echo [2/3] Starting Model Training (CLEFO)
echo ========================================================
echo.

uv run src\main_training\train_clefo.py

echo.
echo Model Training finished.
echo.
echo ========================================================
echo [3/3] Starting Process Optimization (Pyomo)
echo ========================================================
echo.

uv run src\main_optimization\pyomo_nonlinear.py

echo.
echo Process Optimization finished.
echo.
echo ========================================================
echo          All processes completed successfully.
echo ========================================================
echo.

pause