#!/bin/bash
#SBATCH -J teal_shared_triton_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -o logs/teal_shared_triton_sweep_%j.out

mkdir -p logs
mkdir -p results/llama3_shared_triton_sweep

source /oscar/rt/9.2/software/external/miniforge/23.11.0-1/etc/profile.d/conda.sh
conda activate teal

module load cuda/12.9.0-cinr

export SAVE_PATH=/oscar/scratch/rkapoor8/teal_models
export HF_HOME=/oscar/scratch/rkapoor8/hf_cache
export HUGGINGFACE_HUB_CACHE=/oscar/scratch/rkapoor8/hf_cache
export TEAL_BATCH_SPARSE_BACKEND=shared_triton

cd /oscar/home/rkapoor8/TEAL/gpt-fast

for B in 4 8; do
  for S in 0.3 0.4; do
    echo "============================================================"
    echo "RUNNING SPEED: shared_triton batch=${B} sparsity=${S}"
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
      | tee /oscar/home/rkapoor8/TEAL/results/llama3_shared_triton_sweep/shared_triton_b${B}_s${S}_speed_${SLURM_JOB_ID}.txt

    echo "============================================================"
    echo "RUNNING PPL: shared_triton batch=${B} sparsity=${S}"
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
      | tee /oscar/home/rkapoor8/TEAL/results/llama3_shared_triton_sweep/shared_triton_b${B}_s${S}_ppl_${SLURM_JOB_ID}.txt
  done
done
