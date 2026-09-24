@echo off
REM =============================================================================
REM Run 2 LoRA-init fine-tuning jobs SIMULTANEOUSLY on the same GPU.
REM Same dataset + same LLM, different init_weight_approach per job.
REM Each job opens in its own console window so you can watch it live.
REM
REM Only do this if a single run does not use all your VRAM (e.g. ~1/3 each),
REM since both jobs share the same GPU's compute + memory at the same time.
REM =============================================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

set DATASET=cora
set LLM_NAME=smollm2_1.7B
set PEFT_TYPE=lora
set SPLIT=1

REM The two init approaches to run in parallel
set INIT_A=pissa
set INIT_B=orthogonal

echo ==========================================
echo Launching 2 parallel runs
echo   dataset=%DATASET%  llm=%LLM_NAME%  peft=%PEFT_TYPE%  split=%SPLIT%
echo   Job A: init=%INIT_A%
echo   Job B: init=%INIT_B%
echo ==========================================
echo.

start "LoRA-init %INIT_A%" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type %PEFT_TYPE% --init_weight_approach "%INIT_B%" --split %SPLIT% --seed 42

start "LoRA-init %INIT_B%" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%INIT_B%" --split %SPLIT% --seed 43

start "LoRA-init %INIT_B%" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%INIT_B%" --split %SPLIT% --seed 44

start "LoRA-init %INIT_B%" cmd /k python -m train_llm.train_other_lora_init_apporaches --dataset_name "%DATASET%" --llm_name "%LLM_NAME%" --peft_type "%PEFT_TYPE%" --init_weight_approach "%INIT_B%" --split %SPLIT% --seed 45

echo Both jobs launched in separate windows.
echo Each window stays open after finishing (cmd /k) so you can check for errors.
echo Watch VRAM usage with: nvidia-smi -l 2
echo.
pause
