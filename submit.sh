#!/bin/bash

#SBATCH --job-name="graphmae_gpu"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=gmae_run.%j.out
#SBATCH --error=gmae_run.%j.err

# --- ENVIRONMENT SETUP ---
module purge
module load miniconda3
module load cuda/12.1

# Initialize Conda properly for shell scripts
eval "$(conda shell.bash hook)"

# Check for errors
if [ $? -ne 0 ]; then
    echo "ERROR: Modules failed to load."
    exit 1
fi

# --- PATH CONFIGURATION ---
# 1. Directory where you uploaded your GraphMAE2 code
PROJECT_DIR="/scratch/ktanahashi/thesis/GraphMAE2" 

# 2. Name of the environment you created (e.g., 'graphmae')
CONDA_ENV_NAME="graphmae" 

# Activate Environment
conda activate "${CONDA_ENV_NAME}"

# Go to project folder
cd "${PROJECT_DIR}"

# --- CHECKS ---
echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "Python Path: $(which python)"
echo "Python Version: $(python --version)"

# --- RUN COMMAND ---
echo "Starting GraphMAE2 training on Custom Dataset..."

# Optimized Command
python -u main_full_batch.py \
    --dataset custom \
    --device 0 \
    --seed 0 \
    --encoder "gat" \
    --decoder "gat" \
    --mask_method "random" \
    --remask_method "fixed" \
    --mask_rate 0.5 \
    --in_drop 0.2 \
    --attn_drop 0.1 \
    --num_layers 2 \
    --num_dec_layers 1 \
    --num_hidden 512 \
    --num_heads 4 \
    --num_out_heads 1 \
    --max_epoch 200 \
    --max_epoch_f 50 \
    --lr 0.001 \
    --weight_decay 0.04 \
    --loss_fn "sce" \
    --alpha_l 3 \
    --scheduler \
    --linear_prob 

echo "Job finished."