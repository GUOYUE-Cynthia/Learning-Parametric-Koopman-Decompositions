import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

import sys
import json
import os

from koopmanlib.target import VanderPolMathieuTarget

target_dim = 2
param_dim = 1

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

data_path = config['data_settings']['data_path']

n_init = config['data_settings']['n_init']
traj_len = config['data_settings']['traj_len']
mu = config['data_settings']['mu']

k1, k2, k4, w0 = 2,2,1,1

vdp_mathieu = VanderPolMathieuTarget(mu=mu,
                                     n_init=n_init,
                                     traj_len=traj_len,
                                     dim=target_dim,
                                     param_dim=param_dim, 
                                     k1=k1,
                                     k2=k2,
                                     k4=k4,
                                     w0=w0,
                                     seed_x=123, 
                                     seed_param=1)
data_x, data_u = vdp_mathieu.generate_init_data()
data_y = vdp_mathieu.generate_next_data(data_x, data_u)

dict_data = {'data_x': data_x,
              'data_y': data_y,
              'data_u': data_u}

np.save(os.path.join(data_path,'vdpm_data_mu_'+str(mu)+'.npy'), dict_data)

