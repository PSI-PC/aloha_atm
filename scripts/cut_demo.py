import h5py
import numpy as np

FILE = 'episode_1.hdf5'

def cut_demo(images, view):
   # h5_dict = load_h5(FILE)
   # images = h5_dict['observations']['images']
   cut_view = []
   for frames in range(0, 401, 50):
      cut_view.append(images[view][frames : frames + 100])
        
   
   return np.array(cut_view)

def load_h5(file_path):
   # return as a dict.
   def h5_to_dict(h5):
      dict = {}
      for k, v in h5.items():
            if isinstance(v, h5py._hl.group.Group):
               dict[k] = h5_to_dict(v)
            else:
               dict[k] = np.array(v)
      return dict

   with h5py.File(file_path, 'r') as f:
      return h5_to_dict(f)
   
def main():
   h5 = load_h5("./data/preprocessed_demos/train/preprocessed_episode_0.hdf5")
   print(h5['root']['action'].shape)


if __name__ == "__main__":
    main()

