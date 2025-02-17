import numpy as np

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_visible_first


class ATMPretrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        self._index_to_snippet_id = {}
        super().__init__(*args, **kwargs)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("action")
                self.views.remove("task_emb_bert")
                # self.views.remove("extra_states")
                self.views.sort()

            view_key = demo["root"][self.views[0]]

            for snippet_idx in range(0, len(view_key), 2):
                demo_len = demo["root"][self.views[0]][f"snippet_{snippet_idx}"]["video"][0].shape[0]

                if self.cache_all:
                    demo = self.process_demo(demo, snippet_idx)
                    # for v in self.views:
                    #     del demo["root"][v][f"snippet_{snippet_idx}"]["video"]
                    self._cache.append(demo)
                self._demo_id_to_path[demo_idx] = fn
                self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len)})
                # self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len)})
                self._index_to_view_id.update({k: 0 for k in range(start_idx, start_idx + demo_len)})
                self._index_to_snippet_id.update({k: snippet_idx for k in range(start_idx, start_idx + demo_len)})
                self._demo_id_to_start_indices[demo_idx, snippet_idx] = start_idx
                self._demo_id_to_demo_length[demo_idx] = demo_len
                start_idx += demo_len

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        snippet_idx = self._index_to_snippet_id[index]
        snippet_key = f"snippet_{snippet_idx}"
        demo_start_index = self._demo_id_to_start_indices[demo_id, snippet_idx]

        time_offset = index #- demo_start_index

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, snippet_key, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_list_from_disk(demo_id, view, snippet_key, time_offset, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[snippet_idx]
            demo = self.process_demo(self.load_h5(demo_pth), snippet_idx)
            vids = self._load_image_list_from_demo(demo, view, snippet_key, time_offset, backward=True)  # t c h w

        track_start_index = time_offset - demo_start_index
        tracks = demo["root"][view][snippet_key]["tracks"][track_start_index:track_start_index + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view][snippet_key]['vis'][track_start_index:track_start_index + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return vids, tracks, vis, task_emb
