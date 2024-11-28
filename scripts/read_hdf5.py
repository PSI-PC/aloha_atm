import h5py
import numpy as np

FILE = '/home/ws_2425_group_3/development/ATM/data/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5'

# with h5py.File(FILE, 'r') as f:
#    for key in f.keys():
#       print('ACTIONS:')
#       for action in key[0]:
#          print(action)
#       print('OBSERVATIONS:')
#       for obs in key[1]:
#          print(obs)

#    print('ACTIONS')      
#    action_data_set = f['action']
#    print(action_data_set[:10])

#    print('OBSERVATIONS')
#    obs_data_set = f['observations']
#    print(obs_data_set['qpos'][:10].shape)

with h5py.File(FILE, 'r') as f:
   data = f['data']['demo_0']
   print(data['obs']['ee_states'][:5])

def load_h5(fn):
   # return as a dict.
   def h5_to_dict(h5):
      d = {}
      for k, v in h5.items():
            if isinstance(v, h5py._hl.group.Group):
               d[k] = h5_to_dict(v)
            else:
               d[k] = np.array(v)
      return d

   with h5py.File(fn, 'r') as f:
      return h5_to_dict(f)
   
#print(load_h5('/home/ws_2425_group_3/development/ATM/data/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5'))

