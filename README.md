# ALOHA ATM
This is the documentation for our praktikum and seminar "Deep Learning in Robotics".

## Setup
- clone the repo
- cd aloha_atm
- conda create -f environment.yml
- conda activate aloha_atm
- python3 scripts/preprocess_aloha.py

## Changes from original ATM
- cotracker3 instead of cotracker2 (loading issues)

## IMPORTANT
- don't touch libero folder

## PROCEDURE
- preprocess_aloha
- split_aloha_dataset
- train_aloha_track_transformer
- train_aloha_policy
- eval_policy_network / merge_videos