@echo off
REM Neural Network Analysis - Full Pipeline Runner (Windows)
REM This script runs the complete 3-stage analysis pipeline

setlocal enabledelayedexpansion

REM Configuration
set MODEL_PATH=%1
if "%MODEL_PATH%"=="" set MODEL_PATH=models/backdoored_resnet18.pth

set CLEAN_MODEL_PATH=%2
if "%CLEAN_MODEL_PATH%"=="" set CLEAN_MODEL_PATH=models/clean_resnet18.pth

set ARCHITECTURE=%3
if "%ARCHITECTURE%"=="" set ARCHITECTURE=resnet18

set OUTPUT_DIR=%4
if "%OUTPUT_DIR%"=="" (
    for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do set OUTPUT_DIR=results/pipeline_%%c%%a%%b_%%d%%e%%f
)

echo ==========================================
echo Neural Network Analysis - Full Pipeline
echo ==========================================
echo Model: %MODEL_PATH%
echo Clean Model: %CLEAN_MODEL_PATH%
echo Architecture: %ARCHITECTURE%
echo Output Directory: %OUTPUT_DIR%
echo ==========================================

REM Create output directory
mkdir "%OUTPUT_DIR%" 2>nul

REM Stage 1: TSS Extraction
echo.
echo Stage 1: TSS Extraction
echo ------------------------
cd TSS

echo Extracting TSS from backdoored model...
python activation_topotroj.py ^
    --model_type backdoored ^
    --input_type clean ^
    --model_path "../%MODEL_PATH%" ^
    --architecture "%ARCHITECTURE%" ^
    --run_topology ^
    --output_dir "../%OUTPUT_DIR%/tss_backdoored" ^
    --batch_size 32 ^
    --sample_limit 2000

echo Extracting TSS from backdoored model with triggered inputs...
python activation_topotroj.py ^
    --model_type backdoored ^
    --input_type triggered ^
    --model_path "../%MODEL_PATH%" ^
    --architecture "%ARCHITECTURE%" ^
    --run_topology ^
    --trigger_pattern_size 3 ^
    --trigger_pixel_value 1.0 ^
    --trigger_location br ^
    --poison_target_label 0 ^
    --output_dir "../%OUTPUT_DIR%/tss_backdoored_triggered" ^
    --batch_size 32 ^
    --sample_limit 2000

echo Extracting TSS from clean model...
python activation_topotroj.py ^
    --model_type clean ^
    --input_type clean ^
    --model_path "../%CLEAN_MODEL_PATH%" ^
    --architecture "%ARCHITECTURE%" ^
    --run_topology ^
    --output_dir "../%OUTPUT_DIR%/tss_clean" ^
    --batch_size 32 ^
    --sample_limit 2000

cd ..

REM Stage 2: GNN Training
echo.
echo Stage 2: GNN Training
echo ---------------------
cd GNN

REM Check if TSS data exists
set BACKDOORED_TSS="../%OUTPUT_DIR%/tss_backdoored/backdoored_model_clean_inputs_topological_analysis.json"
set CLEAN_TSS="../%OUTPUT_DIR%/tss_clean/clean_model_clean_inputs_topological_analysis.json"

if not exist "!BACKDOORED_TSS!" (
    echo Warning: TSS data not found. Using existing TSS data for training...
    set BACKDOORED_TSS="../data/Efficient Net TSS backdoored.csv"
    set CLEAN_TSS="../data/Efficient Net TSS clean.csv"
)

echo Training GNN with TSS data...
python main.py ^
    --train_backdoored "!BACKDOORED_TSS!" ^
    --train_clean "!CLEAN_TSS!" ^
    --architecture "%ARCHITECTURE%" ^
    --model_type gnn ^
    --epochs 50 ^
    --lr 0.001 ^
    --batch_size 32 ^
    --hidden_dim 64 ^
    --output_dir "../%OUTPUT_DIR%/gnn_models"

echo Running cross-architecture evaluation...
python train_and_cross_eval_gnn.py ^
    --train_arch "%ARCHITECTURE%" ^
    --output_dir "../%OUTPUT_DIR%/gnn_cross_eval"

cd ..

REM Stage 3: ASR Analysis
echo.
echo Stage 3: ASR Analysis
echo ---------------------
cd ASR_ANALYSES\tss_comparison

REM Setup environment if not already done
if not exist "setup_done.flag" (
    echo Setting up TSS comparison environment...
    pip install -e . >nul 2>&1
    echo. > setup_done.flag
)

echo Running GNN-based ASR analysis...
python scripts/gnn_prediction_analysis.py ^
    --model_path "../../%MODEL_PATH%" ^
    --gnn_path "../../%OUTPUT_DIR%/gnn_models/gnn_model.pth" ^
    --device cuda ^
    --removal_ratio 0.15 ^
    --poison_ratio 0.1 ^
    --save_results "../../%OUTPUT_DIR%/gnn_asr_analysis.json"

echo Running ground truth ASR analysis...
python scripts/ground_truth_analysis.py ^
    --model_path "../../%MODEL_PATH%" ^
    --tss_data "../../data/Efficient Net TSS backdoored.csv" ^
    --removal_ratio 0.15 ^
    --poison_ratio 0.1 ^
    --save_results "../../%OUTPUT_DIR%/ground_truth_asr_analysis.json"

cd ..\..

REM Generate Summary Report
echo.
echo Generating Summary Report
echo -------------------------

(
echo # Pipeline Analysis Summary
echo.
echo **Date**: %date% %time%
echo **Model**: %MODEL_PATH%
echo **Clean Model**: %CLEAN_MODEL_PATH%
echo **Architecture**: %ARCHITECTURE%
echo.
echo ## Stage 1: TSS Extraction
echo - Backdoored model ^(clean inputs^): `%OUTPUT_DIR%/tss_backdoored/`
echo - Backdoored model ^(triggered inputs^): `%OUTPUT_DIR%/tss_backdoored_triggered/`
echo - Clean model: `%OUTPUT_DIR%/tss_clean/`
echo.
echo ## Stage 2: GNN Training
echo - Trained models: `%OUTPUT_DIR%/gnn_models/`
echo - Cross-evaluation: `%OUTPUT_DIR%/gnn_cross_eval/`
echo.
echo ## Stage 3: ASR Analysis
echo - GNN-based analysis: `%OUTPUT_DIR%/gnn_asr_analysis.json`
echo - Ground truth analysis: `%OUTPUT_DIR%/ground_truth_asr_analysis.json`
echo.
echo ## Key Files
echo - `gnn_asr_analysis.json`: GNN-based ASR metrics
echo - `ground_truth_asr_analysis.json`: Ground truth ASR metrics
echo - `gnn_models/`: Trained GNN models and encoders
echo - `tss_*/`: TSS extraction results and topological analysis
echo.
echo ## Next Steps
echo 1. Review ASR analysis results
echo 2. Compare GNN predictions vs ground truth
echo 3. Analyze model pruning effectiveness
echo 4. Generate visualization plots
) > "%OUTPUT_DIR%\pipeline_summary.md"

echo.
echo ==========================================
echo Pipeline Complete!
echo ==========================================
echo Results saved to: %OUTPUT_DIR%
echo.
echo Key Results:
echo - TSS extraction: %OUTPUT_DIR%/tss_*/
echo - GNN models: %OUTPUT_DIR%/gnn_models/
echo - ASR analysis: %OUTPUT_DIR%/*_asr_analysis.json
echo - Summary: %OUTPUT_DIR%/pipeline_summary.md
echo.
echo To view results:
echo   type "%OUTPUT_DIR%\pipeline_summary.md"
echo   python -m json.tool "%OUTPUT_DIR%\gnn_asr_analysis.json"
echo   python -m json.tool "%OUTPUT_DIR%\ground_truth_asr_analysis.json"
echo ==========================================

endlocal
