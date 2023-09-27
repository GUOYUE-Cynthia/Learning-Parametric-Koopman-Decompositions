import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

import sys
import json
import os

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)['fhn_settings']

data_path = config['data_settings']['data_path']

Nx = config['data_settings']['Nx']
n_traj = config['data_settings']['n_traj']
traj_len = config['data_settings']['traj_len']

x = np.linspace(-10,10,Nx)
t = np.arange(0, traj_len, 1)
target_dim = Nx*2
param_dim = 1

from koopmanlib.target import FitzHughNagumoTarget

fhn_pde = FitzHughNagumoTarget(
            x=x,
            dt=1e-5,
            t_step=1e-3,
            dim=target_dim,
            param_dim=param_dim,
            param_input=1e3)

data_z_curr, data_u, data_z_next = fhn_pde.generate_data(n_traj=n_traj,
                                                         traj_len=traj_len,
                                                         seed_z=1,
                                                         seed_param=123)
data_dict = {'data_z_curr': data_z_curr,
              'data_u': data_u,
              'data_z_next': data_z_next}

np.save(os.path.join(data_path, 'data_fhn_Nx_'+str(Nx)+'.npy'), data_dict)
