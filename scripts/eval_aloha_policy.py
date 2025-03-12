from pathlib import Path
import sys

import numpy as np
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import hydra
from omegaconf import DictConfig
import torch
from atm.dataloader import BCDataset, get_dataloader
from atm.policy import *
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange

policy_result_dir_path = "./results/policy/"
policy_result_dir = [d for d in os.listdir(policy_result_dir_path)]
latest_result_dir = policy_result_dir[-1]
model_path = Path(str(policy_result_dir_path + latest_result_dir + "/model_best.ckpt"))

# to edit each time
# model_path = Path("./results/policy/0223_atm-policy_demo1_1605_seed0/model_best.ckpt")
figure_name = "all_action_plots_full_vid_36_cluster_act_zwischenstand.png"

@hydra.main(config_path="../conf/train_bc", config_name="libero_vilt_eval.yaml", version_base="1.3")
def evaluate_policy(cfg: DictConfig):
    root_dir = "./data/preprocessed_demos/aloha_lamp/lamp_right_arm/"
    dataset = BCDataset(dataset_dir=glob(os.path.join(root_dir, "eval/")), **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    dataloader = get_dataloader(dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    model.load(model_path)
    model.eval()
    model.reset()

    gt_actions = []
    pred_actions = []

    with torch.no_grad():
        for obs, track_obs, track, task_emb, action, extra_states in tqdm(dataloader):
            # obs = obs[:,:,0:1]
            obs = rearrange(obs, "b v 1 c h w -> b v h w c")
            # obs, track, action = model.preprocess(obs, track, action)
            # pred_action = model.forward(obs, track_obs, track, task_emb, extra_states)
            pred_action, _ = model.act(obs, task_emb, extra_states)
            
            gt_actions.append(action[0, 0, :])
            # pred_actions.append(pred_action[0, 0, :])
            pred_actions.append(pred_action[0, :])

    gt_actions_array = np.array(torch.stack(gt_actions))
    # pred_actions_array = np.array(torch.stack(pred_actions))
    pred_actions_array = np.array(pred_actions)

    t = np.arange(0, len(gt_actions_array[:, 0]))

    num_dimensions = gt_actions_array.shape[1]
    ncols = 5
    nrows = (num_dimensions + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.plot(t, gt_actions_array[:, i], 'b', label='Ground Truth')
        ax.plot(t, pred_actions_array[:, i], 'r', label='Predicted')
        ax.set_ylim(-1, 1.25)
        ax.set_title(f'Action Dimension {i}')
        ax.legend()
    

    for j in range(num_dimensions, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_dir_path = "./plots/"
    save_path = os.path.join(plot_dir_path, figure_name)
    plt.savefig(save_path)
    plt.show()

def main():
    evaluate_policy()

if __name__ == "__main__":
    main()
