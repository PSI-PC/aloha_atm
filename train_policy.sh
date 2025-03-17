#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

module load devel/miniconda

conda activate aloha_atm

export WANDB_API_KEY="c44cc0218f3c141d46861ac42fe23e25ff6745a4"

python3 scripts/train_aloha_policy_atm_two_handed.py
#python3 scripts/train_aloha_policy_atm.py

