import os
import argparse
from glob import glob
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from engine import train_track_transformer as ttt
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig



def get_config():

    # training configs
    CONFIG_NAME = "aloha_track_transformer"

    gpu_ids = [0]

    root_dir = "./data/preprocessed_demos/aloha_lamp/lamp_right_arm/"

    # setup number of epoches and dataset path

    EPOCH = 101
    train_dataset_list = glob(os.path.join(root_dir, "train_1/"))
    val_dataset_list = glob(os.path.join(root_dir, "val_1/"))
    # Manually build a dictionary configuration
    config = {
        'config_name': CONFIG_NAME,  # Replace with your actual config name
        'train_gpus': gpu_ids,  # Example GPU list
        'experiment': f'{CONFIG_NAME}_ep{EPOCH} ',
        'epochs': EPOCH,
        'train_dataset': train_dataset_list,
        'val_dataset': val_dataset_list,
    }
    # Convert to DictConfig (Hydra's configuration object)
    return OmegaConf.create(config)



# command = (f'python -m engine.train_track_transformer --config-name={CONFIG_NAME} '
#            f'train_gpus="{gpu_ids}" '
#            f'experiment={CONFIG_NAME}_ep{EPOCH} '
#            f'epochs={EPOCH} '
#            f'train_dataset="{train_dataset_list}" val_dataset="{val_dataset_list}" ')

# os.system(command)




# @hydra.main(config_path="../conf/train_track_transformer", version_base="1.3")
def main():
    # environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["HYDRA_FULL_ERROR"] = "1"



    # Convert to DictConfig before passing to train_track_transformer.main()
    # config = get_config() # config override epochs etc, not hydra config
    # print(cfg.keys())
    # Call the main function from train_track_transformer

    # import yaml
    # with open("conf/train_track_transformer/aloha_track_transformer.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    # print(config)
    ttt.main()

if __name__ == "__main__":
    CONFIG_NAME = "aloha_track_transformer"
    gpu_ids = [0]
    root_dir = "./data/preprocessed_demos/aloha_lamp/lamp_right_arm/"
    # setup number of epoches and dataset path
    EPOCH = 4004
    train_dataset_list = glob(os.path.join(root_dir, "eval/"))
    val_dataset_list = glob(os.path.join(root_dir, "eval/"))
    
    sys.argv = [
        'train_aloha_track_transformer.py', 
        f'--config-name={CONFIG_NAME}',
        f'train_gpus={gpu_ids}',
        f'experiment={CONFIG_NAME}_ep{EPOCH}',
        f'epochs={EPOCH}',
        f'train_dataset={train_dataset_list}',
        f'val_dataset={val_dataset_list}'
    ]

    main()



