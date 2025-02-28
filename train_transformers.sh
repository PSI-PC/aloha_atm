#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=08:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1

module load devel/miniconda

conda env create -f environment.yml
conda activate aloha_atm

python3 scripts/split_aloha_dataset.py
python3 scripts/train_aloha_track_transformer.py
python3 scripts/train_aloha_policy_atm.py
python3 scripts/eval_aloha_policy.py