#!/bin/bash
#SBATCH -J teal_aggressive_fastcheck
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH -t 06:00:00
#SBATCH -o logs/teal_aggressive_fastcheck_%j.out

mkdir -p logs
mkdir -p results/llama3_aggressive_fastcheck

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export SAVE_PATH=/oscar/scratch/rkapoor8/teal_models
export HF_HOME=/oscar/scratch/rkapoor8/hf_cache
export HUGGINGFACE_HUB_CACHE=/oscar/scratch/rkapoor8/hf_cache
export TEAL_BATCH_SPARSE_BACKEND=shared_triton

cd /oscar/home/rkapoor8/TEAL/gpt-fast

# B BB sparsity MIN_ACTIVE
CONFIGS=(
  "8 16 0.8 1"
  "8 16 0.8 2"
  "8 16 0.8 4"

  "16 16 0.8 1"
  "16 16 0.8 2"
  "16 16 0.8 4"

  "32 32 0.8 1"
  "32 32 0.8 2"
  "32 32 0.8 4"
)

for CFG in "${CONFIGS[@]}"; do
  read B BB S MIN_ACTIVE <<< "$CFG"

  export TEAL_BLOCK_B=${BB}
  export TEAL_BLOCK_N=128
  export TEAL_BLOCK_D=64
  export TEAL_BLOCK_MIN_ACTIVE=${MIN_ACTIVE}

  echo "============================================================"
  echo "SPEED: B=${B} BB=${BB} sparsity=${S} MIN_ACTIVE=${MIN_ACTIVE}"
  echo "============================================================"

  rm -rf /tmp/torchinductor_rkapoor8
  rm -rf ~/.triton/cache

  CUDA_VISIBLE_DEVICES=0 python generate.py \
    --compile \
    --checkpoint_path $SAVE_PATH/meta-llama/Meta-Llama-3-8B/model.pth \
    --hist_path ../models/Llama-3-8B/histograms \
    --sparsity ${S} \
    --max_new_tokens 200 \
    --batch_size ${B} \
    | tee /oscar/home/rkapoor8/TEAL/results/llama3_aggressive_fastcheck/b${B}_s${S}_min${MIN_ACTIVE}_speed_${SLURM_JOB_ID}.txt

  echo "============================================================"
  echo "PPL: B=${B} BB=${BB} sparsity=${S} MIN_ACTIVE=${MIN_ACTIVE}"
  echo "============================================================"

  rm -rf /tmp/torchinductor_rkapoor8
  rm -rf ~/.triton/cache

  CUDA_VISIBLE_DEVICES=0 python eval_ppl_decode.py \
    --checkpoint_path $SAVE_PATH/meta-llama/Meta-Llama-3-8B/model.pth \
    --hist_path ../models/Llama-3-8B/histograms \
    --sparsity ${S} \
    --batch_size ${B} \
    --seq_len 128 \
    --max_batches 8 \
    | tee /oscar/home/rkapoor8/TEAL/results/llama3_aggressive_fastcheck/b${B}_s${S}_min${MIN_ACTIVE}_ppl_${SLURM_JOB_ID}.txt
done