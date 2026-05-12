#!/bin/bash
#SBATCH -J teal_mistral_dense
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/teal_mistral_dense_%j.out

mkdir -p logs
mkdir -p results/mistral_baseline

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export TMPDIR=/oscar/home/rkapoor8/tmp
mkdir -p $TMPDIR

export SAVE_PATH=/oscar/home/rkapoor8/teal_models

cd /oscar/home/rkapoor8/TEAL/gpt-fast

CUDA_VISIBLE_DEVICES=0 python generate.py \
  --compile \
  --checkpoint_path $SAVE_PATH/mistralai/Mistral-7B-v0.1/model.pth \
  --max_new_tokens 200 \
  | tee /oscar/home/rkapoor8/TEAL/results/mistral_baseline/dense_${SLURM_JOB_ID}.txt