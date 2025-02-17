import numpy as np
import torch

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_nearest_to_grids


class BCDataset(BaseDataset):
    def __init__(self, track_obs_fs=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.track_obs_fs = track_obs_fs

    def __getitem__(self, index):
        index = index # mal 10
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        snippet_idx = self._index_to_snippet_id[index]
        snippet_key = f"snippet_{snippet_idx}"
        demo_start_index = self._demo_id_to_start_indices[demo_id, snippet_idx]

        time_offset = index #- demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                if self.cache_image:
                    all_view_frames.append(self._load_image_list_from_demo(demo, view, snippet_key, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_demo(demo, view, snippet_key, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
                else:
                    all_view_frames.append(self._load_image_list_from_disk(demo_id, view, snippet_key, time_offset))  # t c h w
                    all_view_track_transformer_frames.append(
                        torch.stack([self._load_image_list_from_disk(demo_id, view, snippet_key, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                    )  # t tt_fs c h w
        else:
            demo_pth = self._demo_id_to_path[snippet_idx]
            demo = self.process_demo(self.load_h5(demo_pth), snippet_idx)
            all_view_frames = []
            all_view_track_transformer_frames = []
            for view in self.views:
                all_view_frames.append(self._load_image_list_from_demo(demo, view, snippet_key, time_offset))  # t c h w
                all_view_track_transformer_frames.append(
                    torch.stack([self._load_image_list_from_demo(demo, view, snippet_key, time_offset + t, num_frames=self.track_obs_fs, backward=True) for t in range(self.frame_stack)])
                )  # t tt_fs c h w

        all_view_tracks = []
        all_view_vis = []
        for view in self.views:
            all_time_step_tracks = []
            all_time_step_vis = []
            # if(time_offset == 102):
            #     print(1)
            for track_start_index in range(time_offset - demo_start_index, time_offset+self.frame_stack - demo_start_index):
                all_time_step_tracks.append(demo["root"][view][snippet_key]["tracks"][track_start_index:track_start_index + self.num_track_ts])  # track_len n 2
                all_time_step_vis.append(demo["root"][view][snippet_key]['vis'][track_start_index:track_start_index + self.num_track_ts])  # track_len n
            all_view_tracks.append(torch.stack(all_time_step_tracks, dim=0))
            all_view_vis.append(torch.stack(all_time_step_vis, dim=0))

        obs = torch.stack(all_view_frames, dim=0)  # v t c h w
        track = torch.stack(all_view_tracks, dim=0)  # v t track_len n 2
        vi = torch.stack(all_view_vis, dim=0)  # v t track_len n
        track_transformer_obs = torch.stack(all_view_track_transformer_frames, dim=0)  # v t tt_fs c h w

        # augment rgbs and tracks
        if np.random.rand() < self.aug_prob:
            obs, track = self.augmentor((obs / 255., track))
            obs = obs * 255.

        # sample tracks
        sample_track, sample_vi = [], []
        for i in range(len(self.views)):
            sample_track_per_time, sample_vi_per_time = [], []
            for t in range(self.frame_stack):
                track_i_t, vi_i_t = sample_tracks_nearest_to_grids(track[i, t], vi[i, t], num_samples=self.num_track_ids)
                sample_track_per_time.append(track_i_t)
                sample_vi_per_time.append(vi_i_t)
            sample_track.append(torch.stack(sample_track_per_time, dim=0))
            sample_vi.append(torch.stack(sample_vi_per_time, dim=0))
        track = torch.stack(sample_track, dim=0)
        vi = torch.stack(sample_vi, dim=0)

        actions = demo["root"]["action"][time_offset:time_offset + self.frame_stack]
        task_embs = demo["root"]["task_emb_bert"]
        # extra_states = {k: v[time_offset:time_offset + self.frame_stack] for k, v in
        #                 demo['root']['extra_states'].items()}
        extra_states = {}

        return obs, track_transformer_obs, track, task_embs, actions, extra_states
