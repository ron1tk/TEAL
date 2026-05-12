#!/bin/bash
#SBATCH -J teal_llama3_ppl_sparse_b1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/teal_llama3_ppl_sparse_b2_%j.out

mkdir -p logs
mkdir -p results/llama3_ppl

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export SAVE_PATH=/oscar/scratch/rkapoor8/teal_models
export HF_HOME=/oscar/scratch/rkapoor8/hf_cache
export HUGGINGFACE_HUB_CACHE=/oscar/scratch/rkapoor8/hf_cache

cd /oscar/home/rkapoor8/TEAL/gpt-fast

CUDA_VISIBLE_DEVICES=0 python eval_ppl_decode.py \
  --checkpoint_path /oscar/scratch/rkapoor8/teal_models/meta-llama/Meta-Llama-3-8B/model.pth \
  --hist_path ../models/Llama-3-8B/histograms \
  --sparsity 0.5 \
  --batch_size 1 \
  --seq_len 128 \
  --max_batches 8
  | tee /oscar/home/rkapoor8/TEAL/results/llama3_ppl/sparse_b2_ppl_${SLURM_JOB_ID}.txt