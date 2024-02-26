#!/bin/bash
#
#SBATCH --job-name=mae-gen
#SBATCH --account=jamiemmt
#SBATCH --partition=gpu-a100
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --requeue
#
#SBATCH --open-mode=truncate
#SBATCH --chdir=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src
#SBATCH --output=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/mae/out.log
#SBATCH --error=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/mae/out.err

export PATH=$PATH:$HOME/miniconda3/bin

echo "---------start-----------"

python /gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/mae/cifar100.py --batch_size 256
