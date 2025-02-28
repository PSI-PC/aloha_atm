#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=08:00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1

module load devel/miniconda

conda env create -f environment.yml
conda activate aloha_atm

export WANDB_API_KEY="c44cc0218f3c141d46861ac42fe23e25ff6745a4"

python3 scripts/split_aloha_dataset.py
python3 scripts/train_aloha_track_transformer.py
python3 scripts/train_aloha_policy_atm.py
python3 scripts/eval_aloha_policy.py