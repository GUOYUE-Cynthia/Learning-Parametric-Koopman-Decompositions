import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

import sys
import json
import os

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

data_path = config['data_settings']['data_path']

Nx = config['data_settings']['Nx']
n_init = config['data_settings']['n_init']
traj_len = config['data_settings']['traj_len']
forcing_type = config['data_settings']['forcing_type']

x = np.linspace(-10,10,Nx)
t = np.arange(0, traj_len, 1)
target_dim = Nx*2
param_dim = 1

from koopmanlib.target import KortewegDeVriesTarget

# Set the size of the domain, and create the discretized grid.
L = 2 * np.pi
Nx = 128
dx = L / (Nx - 1.0)
x = np.linspace(-np.pi, np.pi, Nx)
T = 0.01

def v_func(x, c):
    return np.exp(-25 * (x - c)**2)

c1, c2, c3 = -np.pi/2, 0, np.pi/2
v1 = v_func(x, c1).reshape(1,-1)
v2 = v_func(x, c2).reshape(1,-1)
v3 = v_func(x, c3).reshape(1,-1)

v_list = np.concatenate([v1,v2,v3], axis=0)

umax = 1
umin = -umax

target_dim = Nx
param_dim = 3

kdv = KortewegDeVriesTarget(n_init=n_init,
            traj_len=traj_len,
            x=x,
            t_step=T,
            dim=Nx,
            param_dim=param_dim,
            forcing_type=forcing_type,
            v_list=v_list,
            L=L)

np.random.seed(123)
seed_IC = np.random.randint(0,100,size=(n_init,))

y0_list = []
for seed in seed_IC:
    y0 = kdv.generate_y0(seed)
    y0_list.append(y0)
y0_list = np.asarray(y0_list)

param_list_group = np.random.uniform(low=0, high=1, size=(n_init, traj_len, param_dim)) * (umax - umin) + umin

soln_outer_list = []
for y0, param_list in zip(y0_list, param_list_group):
    # Calculate inner solution for each y0 and param_list (for one trajectory)
    soln_inner_list = [y0]
    for param in param_list:
        soln = kdv.kdv_solution(y0, T, param)
        y0 = soln.y.T[-1]
        soln_inner_list.append(y0)

    soln_inner_list = np.asarray(soln_inner_list)
    
    soln_outer_list.append(soln_inner_list)
    
soln_outer_list = np.asarray(soln_outer_list)

data_x = soln_outer_list[:,:-1,:].reshape(-1, target_dim)
data_y = soln_outer_list[:,1:,:].reshape(-1, target_dim)
data_u = param_list_group.reshape(-1,param_dim)

data_dict = {'data_x': data_x,
             'data_y': data_y,
             'data_u': data_u}

np.save(os.path.join(data_path, 'data_kdv_'+forcing_type+'.npy'), data_dict)
