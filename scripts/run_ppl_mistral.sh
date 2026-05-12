#!/bin/bash
#SBATCH -J teal_ppl_mistral
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/teal_ppl_mistral_%j.out

mkdir -p logs
mkdir -p results/mistral_baseline

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export TMPDIR=/oscar/home/rkapoor8/tmp
mkdir -p $TMPDIR

cd /oscar/home/rkapoor8/TEAL

export TEAL_PATH=/oscar/home/rkapoor8/TEAL/models/Mistral-7B

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python teal/ppl_test.py \
  --model_name /oscar/home/rkapoor8/teal_models/mistralai/Mistral-7B-v0.1 \
  --teal_path /oscar/home/rkapoor8/TEAL/models/Mistral-7B \
  --sparsity 0.5