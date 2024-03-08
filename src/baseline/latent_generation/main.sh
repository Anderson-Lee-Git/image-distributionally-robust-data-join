#!/bin/bash
#
#SBATCH --job-name=latent-gen
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --requeue
#
#SBATCH --open-mode=truncate
#SBATCH --chdir=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src
#SBATCH --output=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/baseline/latent_generation/out.log
#SBATCH --error=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/baseline/latent_generation/out.err

export PATH=$PATH:$HOME/miniconda3/bin

python baseline/latent_generation/main.py \
--output_dir="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/baseline/latent_generation" \
--log_dir="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/baseline/latent_generation" \
--model=resnet50 \
--batch_size=256 \
--input_size=224 \
--num_workers=5 \
--data_subset=1.0 \
--dataset=celebA \
--data_group=0 \
--num_classes=2 \
--unbalanced \
--ckpt="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/resnet50_baseline_unbalanced_curated_celebA/winter-sweep-3/checkpoint-9.pth"

rm -r "/scr/lee0618/done_$dataset"
