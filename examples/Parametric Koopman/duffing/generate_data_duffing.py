from koopmanlib.target import DuffingParamTarget
import numpy as np
import json
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

data_path = config['data_settings']['data_path']
n_total_data = config['data_settings']['n_total_data']

# Number of the trajectories for each choice of parameters
n_traj_per_param = config['data_settings']['n_traj_per_param']

# Number of the points on each trajectory
traj_len = config['data_settings']['traj_len']

seed_x = config['data_settings']['seed_x']
seed_param = config['data_settings']['seed_param']


# Number of choices of parameters
n_param = int(n_total_data / n_traj_per_param)


print('n_traj_per_param', n_traj_per_param)
print('n_param', n_param)

duffing_param = DuffingParamTarget(dim=2, param_dim=3)

data_x, data_u = duffing_param.generate_init_data(n_param=n_param,
                                                  traj_len=traj_len,
                                                  n_traj_per_param=n_traj_per_param,
                                                  seed_x=seed_x,
                                                  seed_param=seed_param)

data_y = duffing_param.generate_next_data(data_x, data_u)

position = traj_len * n_traj_per_param * np.arange(0, n_param+1)

data_u_sep = [data_u[start:end]
              for start, end in zip(position[:-1], position[1:])]
data_u_sep = np.asarray(data_u_sep)

data_x_sep = [data_x[start:end]
              for start, end in zip(position[:-1], position[1:])]
data_x_sep = np.asarray(data_x_sep)

data_y_sep = [data_y[start:end]
              for start, end in zip(position[:-1], position[1:])]
data_y_sep = np.asarray(data_y_sep)


# Generate data
dict_data = {'data_x': data_x,
             'data_y': data_y,
             'data_u': data_u}

np.save(os.path.join(data_path, 'duffing_data_n_param_'+str(n_param)+'_len_' +
        str(traj_len)+'_n_traj_per_'+str(n_traj_per_param)+'.npy'), dict_data)

dict_data_sep = {'data_x_sep': data_x_sep,
                 'data_y_sep': data_y_sep,
                 'data_u_sep': data_u_sep}

np.save(os.path.join(data_path, 'duffing_data_sep_n_param_'+str(n_param)+'_len_' +
        str(traj_len)+'_n_traj_per_'+str(n_traj_per_param)+'.npy'), dict_data_sep)
