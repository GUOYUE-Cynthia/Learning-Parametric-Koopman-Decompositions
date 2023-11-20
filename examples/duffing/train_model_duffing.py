
from tensorflow.keras.optimizers import Adam
from koopmanlib.param_solver import KoopmanParametricDLSolver
from koopmanlib.solver import KoopmanDLSolver
from koopmanlib.target import DuffingParamTarget
from koopmanlib.dictionary import PsiNN
import tensorflow as tf
import numpy as np
import json
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from koopmanlib.K_structure import Model_K_u_Layer


config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)


data_path = config['data_settings']['data_path']

weights_path = config['nn_settings']['weights_path']

n_total_data = config['data_settings']['n_total_data']

# Number of the trajectories for each choice of parameters
n_traj_per_param = config['data_settings']['n_traj_per_param']

# Number of choices of parameters
n_param = int(n_total_data / n_traj_per_param)
traj_len = config['data_settings']['traj_len']
dict_layer_size = config['nn_settings']['dict_layer_size']
n_psi_train = config['nn_settings']['n_psi_train']

# Train EDMD-DL for different parametric cases
# Load data
dict_data_sep = np.load(os.path.join(data_path, 'duffing_data_sep_n_param_'+str(n_param) +
                        '_len_'+str(traj_len)+'_n_traj_per_'+str(n_traj_per_param)+'.npy'), allow_pickle=True)

data_x_sep = dict_data_sep[()]['data_x_sep']
data_y_sep = dict_data_sep[()]['data_y_sep']
data_u_sep = dict_data_sep[()]['data_u_sep']

target_dim = 2
param_dim = 3

n_psi = 1 + target_dim + n_psi_train

edmd_epochs = config['nn_settings']['edmd_epochs']

for i in range(data_x_sep.shape[0]):
    data_x_train = data_x_sep[i]
    data_y_train = data_y_sep[i]
    data_u_train = data_u_sep[i]

    data_x_valid = data_x_train
    data_y_valid = data_y_train

    data_train = [data_x_train, data_y_train]
    data_valid = [data_x_valid, data_y_valid]

    basis_function = PsiNN(layer_sizes=dict_layer_size,
                           n_psi_train=n_psi_train)

    solver = KoopmanDLSolver(dic=basis_function,
                             target_dim=2,
                             reg=0.1)

    solver.build(data_train=data_train,
                 data_valid=data_valid,
                 epochs=edmd_epochs,
                 batch_size=1000,
                 lr=1e-4,
                 log_interval=20,
                 lr_decay_factor=0.8)

    solver.model.save_weights(os.path.join(weights_path, 'edmd_duffing_weights_data_'+str(
        i)+'_n_traj_per_param_'+str(n_traj_per_param)+'_n_param_'+str(n_param)+'.h5'))

# Train PK-NN

# Load data

dict_data = np.load(os.path.join(data_path, 'duffing_data_n_param_'+str(n_param) +
                    '_len_'+str(traj_len)+'_n_traj_per_'+str(n_traj_per_param)+'.npy'), allow_pickle=True)

data_x = dict_data[()]['data_x']
data_y = dict_data[()]['data_y']
data_u = dict_data[()]['data_u']

dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

K_layer_size = config['nn_settings']['K_layer_size']
pknn_epochs = config['nn_settings']['pknn_epochs']

model_K_u = Model_K_u_Layer(layer_sizes=K_layer_size, 
                                n_psi=n_psi)

solver_pk = KoopmanParametricDLSolver(
    target_dim=target_dim, 
    param_dim=param_dim, 
    n_psi=n_psi, 
    dic=dic_pk, 
    model_K_u=model_K_u)

model_pk, model_K_u_pred_pk = solver_pk.generate_model()

zeros_data_y_train = tf.zeros_like(dic_pk(data_y))

model_pk.compile(optimizer=Adam(0.001),
                 loss='mse')

lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                    factor=0.1,
                                                    patience=50,
                                                    verbose=0,
                                                    mode='auto',
                                                    min_delta=0.0001,
                                                    cooldown=0,
                                                    min_lr=1e-12)

# es_callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss',
#                                                 min_delta=0,
#                                                 patience=20,
#                                                 verbose=0,
#                                                 mode='min',
#                                                 baseline=1e-7)

history = model_pk.fit(x=[data_x, data_y, data_u],
                       y=zeros_data_y_train,
                       epochs=pknn_epochs,
                       batch_size=200,
                       # callbacks=[lr_callbacks,es_callbacks],
                       callbacks=[lr_callbacks],
                       verbose=1)

model_pk.save_weights(os.path.join(weights_path, 'pk_duffing_weights_data_' +
                      str(n_traj_per_param)+'_n_param_'+str(n_param)+'.h5'))
