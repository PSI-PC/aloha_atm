#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pascal.siekiera@gmx.de

module load devel/miniconda

# conda env create -f environment.yml
conda activate aloha_atm

# pip install -e third_party/robosuite/
# pip install -e third_party/robomimic/

export WANDB_API_KEY="c44cc0218f3c141d46861ac42fe23e25ff6745a4"

# python3 scripts/split_aloha_dataset.py
# python3 scripts/train_aloha_track_transformer_two_handed.py
python3 scripts/train_aloha_policy_atm_two_handed.py
# python3 scripts/eval_aloha_policy.py
