import json
import os
import sys

import numpy as np
import tensorflow as tf

from koopmanlib.target import KortewegDeVriesTarget

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


config_file = sys.argv[1]

with open(config_file) as f:
    # config = json.load(f)['linear'] # # Use this if we want to generate data for the sin example
    config = json.load(f)['sin'] # Use this if we want to generate data for the sin example


data_path = config["data_settings"]["data_path"]


# Check if the folder exists, if not, create it
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Directory {data_path} created.")
else:
    print(f"Directory {data_path} already exists.")

Nx = config["data_settings"]["Nx"]
n_traj = config["data_settings"]["n_traj"]
traj_len = config["data_settings"]["traj_len"]
forcing_type = config["data_settings"]["forcing_type"]


# Set the size of the domain, and create the discretized grid.
L = 2 * np.pi
Nx = 128
dx = L / (Nx - 1.0)
x = np.linspace(-np.pi, np.pi, Nx)
T = 0.01


def v_func(x, c):
    return np.exp(-25 * (x - c) ** 2)


c1, c2, c3 = -np.pi / 2, 0, np.pi / 2
v1 = v_func(x, c1).reshape(1, -1)
v2 = v_func(x, c2).reshape(1, -1)
v3 = v_func(x, c3).reshape(1, -1)

v_list = np.concatenate([v1, v2, v3], axis=0)

umax = 1
umin = -umax

target_dim = Nx
param_dim = 3

kdv = KortewegDeVriesTarget(
    x=x, t_step=T, dim=Nx, param_dim=param_dim, forcing_type=forcing_type, v_list=v_list, L=L
)

data_x, data_y, data_u = kdv.generate_data(
    n_traj=n_traj, traj_len=traj_len, seed_y0=123, seed_param=1
)

data_dict = {"data_x": data_x, "data_y": data_y, "data_u": data_u}

np.save(os.path.join(data_path, 'data_kdv_'+forcing_type+'.npy'), data_dict)
