#!/bin/bash
#
#SBATCH --job-name=baseline-unb-celebA
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
#SBATCH --output=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/resnet50_baseline_unbalanced_curated_celebA/out.log
#SBATCH --error=/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/resnet50_baseline_unbalanced_curated_celebA/out.err

dataset="celebA_v1.0"
start_time=$(date +%s)
if [ ! -d "/scr/lee0618/$dataset" ]
then
    mkdir /scr/lee0618/ && echo 'created /scr/lee0618/'
    echo "Start un-tar $dataset"
    tar xC /scr/lee0618 -f "/gscratch/cse/lee0618/$dataset.tar"

    # Define the directory where you want to start the search
    start_dir="/scr/lee0618"
    # Define the name of the folder you want to find
    folder_name="$dataset"
    # Use the find command to search for the folder
    found_folders=$(find "$start_dir" -type d -name "$folder_name")
    # Check if any folders were found
    if [ -n "$found_folders" ]; then
        echo "Folders found:"
        echo "$found_folders"
        
        # Move each found folder to the start directory
        while IFS= read -r folder; do
            echo "Moving $folder to $start_dir"
            mv "$folder" "$start_dir"
        done <<< "$found_folders"
    else
        echo "No folders found with the name '$folder_name' within '$start_dir' and its subdirectories."
    fi
    echo "In dataset root directory:"
    ls "/scr/lee0618"
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    echo "Take $execution_time seconds to finish un-tar dataset"
    mkdir "/scr/lee0618/done_$dataset"
fi

until [ -d "/scr/lee0618/done_$dataset" ]

do
  sleep 1
done

PROJECT_NAME='resnet50_baseline_unbalanced_curated_celebA'

export PATH=$PATH:$HOME/miniconda3/bin

DATASET_ROOT="/scr/lee0618" python baseline/train.py \
--use_wandb \
--wandb_cont_sweep \
--project_name="$PROJECT_NAME" \
--output_dir="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/$PROJECT_NAME" \
--log_dir="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/logs/$PROJECT_NAME" \
--model=resnet50 \
--batch_size=256 \
--epochs=10 \
--input_size=224 \
--num_workers=5 \
--data_subset=1.0 \
--dataset=celebA \
--data_group=1 \
--num_classes=2 \
--unbalanced

rm -r "/scr/lee0618/done_$dataset"
