#!/bin/bash
#SBATCH -J teal_llama3_sparse_b2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/teal_llama3_sparse_b2_%j.out

mkdir -p logs
mkdir -p results/llama3_baseline

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export SAVE_PATH=/oscar/scratch/rkapoor8/teal_models

cd /oscar/home/rkapoor8/TEAL/gpt-fast

CUDA_VISIBLE_DEVICES=0 python generate.py \
  --compile \
  --checkpoint_path $SAVE_PATH/meta-llama/Meta-Llama-3-8B/model.pth \
  --hist_path ../models/Llama-3-8B/histograms \
  --sparsity 0.5 \
  --max_new_tokens 200 \
  --batch_size 2 \
  | tee /oscar/home/rkapoor8/TEAL/results/llama3_baseline/sparse_b2_probe_${SLURM_JOB_ID}.txt

