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

action_boundaries = np.array([[-0.06749516, -1.0569128,   1.0599808,  -0.159534,   -0.4632622,  -0.23009713, -0.03186427, 
                        -0.2638447,  -0.961806,   -0.01227185, -0.3512816,  -0.40497094, 0.,         -0.09559283],
                     [0.02454369, -0.72557294,  1.2501944,  -0.00920388, -0.1288544,   0.00460194, 0.2671697,   
                        0.36355346,  0.14112623,  1.1658255,   0.27458256,  0.9894176, 0.93112636,  0.7108185]])
def main():
   # max_values = []
   # min_values = []
   # for i in range(36):
   #    h5_file = load_h5(f"./data/preprocessed_demos/aloha_lamp/lamp_right_arm/episode_{i}/preprocessed_episode_{i}.hdf5")
   #    max_values.append(h5_file['root']['action'].max(axis=0))
   #    min_values.append(h5_file['root']['action'].min(axis=0))
   #    print("================MAX================")
   #    print(np.array(max_values).max(axis=0))
   #    print("================MIN================")
   #    print(np.array(min_values).min(axis=0))
   # h5_file = load_h5("../ATM/data/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5")
   # print(h5_file['data']['demo_0']['actions'].var(axis=0))
   h5_file_us = load_h5("./data/preprocessed_demos/aloha_lamp/lamp_right_arm/episode_0/preprocessed_episode_0.hdf5")
   # print((2 * (h5_file_us['root']['action'] - action_boundaries[0])/(action_boundaries[1] - action_boundaries[0]) - 1).var(axis=0))
   print(h5_file_us['root']['action'].shape)


   # print(h5_file['root']['action'].shape)


if __name__ == "__main__":
    main()

