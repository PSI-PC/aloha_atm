import os
import argparse
from glob import glob


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# default track transformer path
# DEFAULT_TRACK_TRANSFORMERS = [
#     "./results/track_transformer/0129_aloha_track_transformer_ep101_1503",
# ]

# get the last generated result folder from track transformer
track_transformer_result_dir_path = "./results/track_transformer/"
track_transformer_result_dir = [d for d in os.listdir(track_transformer_result_dir_path)]
latest_result_dir = track_transformer_result_dir[-1]
DEFAULT_TRACK_TRANSFORMERS = [
    str(track_transformer_result_dir_path + latest_result_dir)
]

# input parameters
# parser = argparse.ArgumentParser()
# parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], 
#                     help="The name of the desired suite, where libero_10 is the alias of libero_long.")
# parser.add_argument("-tt", "--track-transformer", default=None, help="Then path to the trained track transformer.")
# args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_vilt"

train_gpu_ids = [0]
root_dir = "./data/preprocessed_demos/aloha_lamp/lamp_right_arm/"
NUM_DEMOS = 75 #len(next(os.walk(root_dir))[1]) - 3


# suite_name = args.suite
# task_dir_list = os.listdir(os.path.join(root_dir, suite_name))
# task_dir_list.sort()

# dataset
# train_path_list = [f"{root_dir}/{suite_name}/{task_dir}/bc_train_{NUM_DEMOS}" for task_dir in task_dir_list]
# val_path_list = [f"{root_dir}/{suite_name}/{task_dir}/val" for task_dir in task_dir_list]
train_dataset_list = glob(os.path.join(root_dir, "bc_train_20/"))
val_dataset_list = glob(os.path.join(root_dir, "val/"))

# track_fn = DEFAULT_TRACK_TRANSFORMERS[0] # or args.track_transformer
track_fn = './results/track_transformer/0313_aloha_track_transformer_ep1001_0854'

for seed in range(1):
    command = (f'python -m engine.train_bc --config-name={CONFIG_NAME} train_gpus="{train_gpu_ids}" '
                f'experiment=atm-policy_demo{NUM_DEMOS} '
                f'train_dataset="{train_dataset_list}" val_dataset="{val_dataset_list}" '
                f'model_cfg.track_cfg.track_fn={track_fn} '
                f'model_cfg.track_cfg.use_zero_track=False '
                f'model_cfg.spatial_transformer_cfg.use_language_token=False '
                f'model_cfg.temporal_transformer_cfg.use_language_token=False '
                f'seed={seed} ')

    os.system(command)
