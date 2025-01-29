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

@hydra.main(config_path="../conf/train_bc", config_name="libero_vilt_eval.yaml", version_base="1.3")
def evaluate_policy(cfg: DictConfig):

    root_dir = "./data/preprocessed_demos/aloha_lamp/lamp_right_arm/"
    dataset = BCDataset(dataset_dir=glob(os.path.join(root_dir, "test/")), **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    dataloader = get_dataloader(dataset,
                                    mode="val",
                                    num_workers=cfg.num_workers,
                                    batch_size=cfg.batch_size)

    model_path = Path("./results/policy/0127_atm-policy_demo36_1000_seed0/model_best.ckpt")
    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    model.load(model_path)
    model.eval()

    gt_actions = []
    pred_actions = []
    diff_actions = []

    with torch.no_grad():
        for obs, track_obs, track, task_emb, action, extra_states in tqdm(dataloader):
            action_distribution = model.forward(obs, track_obs, track, task_emb, extra_states)
            # print(f"MEAN DEVIATION: {torch.mean(action - action_distribution)}")
            gt_actions.append(action[0, 0, :])
            pred_actions.append(action_distribution[0, 0, :])
            diff_actions.append(action[0, 0, :] - action_distribution[0, 0, :])
            # print(f'gt: ${action[0, 0, :].shape}')
            # print(f'pred: ${action_distribution[0, 0, :].shape}')

    gt_actions_array = np.array(torch.stack(gt_actions))
    pred_actions_array = np.array(torch.stack(pred_actions))
    diff_actions_array = np.array(torch.stack(diff_actions))

    print(action[0, 0, :].shape)
    print(gt_actions_array.shape)


    t = np.arange(0, len(gt_actions_array[:, 0]))

    num_dimensions = gt_actions_array.shape[1]
    ncols = 3
    nrows = (num_dimensions + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.plot(t, gt_actions_array[:, i], 'b', label='Ground Truth')
        ax.plot(t, pred_actions_array[:, i], 'r', label='Predicted')
        # ax.plot(t, diff_actions_array[:, i], 'g--', label='Difference')
        ax.set_title(f'Action Dimension {i}')
        ax.legend()
    

    for j in range(num_dimensions, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("all_action_plots.png")

    plt.show()

    print(np.mean(diff_actions_array))



def main():
    evaluate_policy()

if __name__ == "__main__":
    main()
