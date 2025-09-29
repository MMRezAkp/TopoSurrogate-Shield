@echo off
REM Batch script for running multiple architecture evaluations on Windows
REM This script will run evaluations for different architectures and predictors

echo ============================================================
echo Architecture Evaluation Batch Runner for Windows
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "data_evaluator.py" (
    echo Error: data_evaluator.py not found!
    echo Make sure data_evaluator.py is in the current directory.
    pause
    exit /b 1
)

if not exist "run_evaluation.py" (
    echo Error: run_evaluation.py not found!
    echo Make sure run_evaluation.py is in the current directory.
    pause
    exit /b 1
)

REM Create output directory
if not exist "batch_evaluation_results" mkdir batch_evaluation_results

echo.
echo Available modes:
echo 1. Interactive mode - select train/test CSV files for each evaluation
echo 2. Auto mode - uses predefined CSV file patterns
echo 3. Custom mode - specify train and test CSV files once, run all predictors
echo.

set /p mode="Select mode (1-3): "

if "%mode%"=="1" goto interactive_mode
if "%mode%"=="2" goto auto_mode
if "%mode%"=="3" goto custom_mode

echo Invalid selection. Using interactive mode.
goto interactive_mode

:interactive_mode
echo.
echo ========== INTERACTIVE MODE ==========
echo You will be prompted to select CSV files for each evaluation.
echo.

REM List of architectures to evaluate
set architectures=efficientnet mobilenet mobilenetv2

REM List of predictors to evaluate
set predictors=baseline random_forest xgboost

for %%a in (%architectures%) do (
    echo.
    echo ==========================================
    echo Evaluating architecture: %%a
    echo ==========================================
    
    for %%p in (%predictors%) do (
        echo.
        echo Running evaluation with %%p predictor...
        echo ------------------------------------------
        
        python run_evaluation.py --architecture %%a --predictor %%p --output_dir batch_evaluation_results\%%a_%%p --interactive
        
        if errorlevel 1 (
            echo Error occurred during %%a with %%p predictor
            echo Continuing with next evaluation...
        ) else (
            echo %%a with %%p completed successfully
        )
        
        echo.
        pause
    )
)

goto end

:auto_mode
echo.
echo ========== AUTO MODE ==========
echo Looking for CSV files with standard naming patterns...
echo Expected patterns: *train*.csv, *test*.csv

REM Find train and test files automatically
dir /b *train*.csv >nul 2>&1
if errorlevel 1 (
    echo No training CSV files found with pattern *train*.csv
    echo Please ensure your CSV files follow naming conventions or use custom mode
    pause
    exit /b 1
)

dir /b *test*.csv >nul 2>&1
if errorlevel 1 (
    echo No test CSV files found with pattern *test*.csv
    echo Please ensure your CSV files follow naming conventions or use custom mode
    pause
    exit /b 1
)

REM Get first matching files
for /f %%i in ('dir /b *train*.csv') do set train_file=%%i & goto found_train
:found_train

for /f %%i in ('dir /b *test*.csv') do set test_file=%%i & goto found_test
:found_test

echo Found training file: %train_file%
echo Found test file: %test_file%
echo.

set architectures=efficientnet mobilenet mobilenetv2
set predictors=baseline random_forest xgboost

for %%a in (%architectures%) do (
    echo.
    echo ==========================================
    echo Evaluating architecture: %%a
    echo ==========================================
    
    for %%p in (%predictors%) do (
        echo.
        echo Running %%a with %%p predictor...
        echo ------------------------------------------
        
        python data_evaluator.py --train_csv "%train_file%" --test_csv "%test_file%" --architecture %%a --predictor %%p --output_dir batch_evaluation_results\%%a_%%p
        
        if errorlevel 1 (
            echo Error occurred during %%a with %%p predictor
        ) else (
            echo %%a with %%p completed successfully
        )
    )
)

goto end

:custom_mode
echo.
echo ========== CUSTOM MODE ==========
echo Please specify the train and test CSV files to use for all evaluations.

set /p train_csv="Enter path to training CSV file: "
set /p test_csv="Enter path to test CSV file: "

if not exist "%train_csv%" (
    echo Error: Training file not found: %train_csv%
    pause
    exit /b 1
)

if not exist "%test_csv%" (
    echo Error: Test file not found: %test_csv%
    pause
    exit /b 1
)

echo.
echo Using training file: %train_csv%
echo Using test file: %test_csv%
echo.

set architectures=efficientnet mobilenet mobilenetv2
set predictors=baseline random_forest xgboost

for %%a in (%architectures%) do (
    echo.
    echo ==========================================
    echo Evaluating architecture: %%a
    echo ==========================================
    
    for %%p in (%predictors%) do (
        echo.
        echo Running %%a with %%p predictor...
        echo ------------------------------------------
        
        python data_evaluator.py --train_csv "%train_csv%" --test_csv "%test_csv%" --architecture %%a --predictor %%p --output_dir batch_evaluation_results\%%a_%%p
        
        if errorlevel 1 (
            echo Error occurred during %%a with %%p predictor
        ) else (
            echo %%a with %%p completed successfully
        )
    )
)

:end
echo.
echo ============================================================
echo BATCH EVALUATION COMPLETED
echo ============================================================
echo Results saved in: batch_evaluation_results\
echo.

REM Show summary of results
if exist "batch_evaluation_results" (
    echo Generated result directories:
    dir /b batch_evaluation_results
    echo.
    echo Each directory contains:
    echo - *_results.csv: Main results with predictions
    echo - *_metrics.csv: Evaluation metrics (RMSE, MAE, correlation)
    echo - *.pkl files: Saved preprocessors
)

echo.
echo Press any key to exit...
pause >nul

