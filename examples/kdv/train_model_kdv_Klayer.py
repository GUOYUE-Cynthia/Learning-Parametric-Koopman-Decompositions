from koopmanlib.param_solver import KoopmanParametricDLSolver, KoopmanLinearDLSolver, KoopmanBilinearDLSolver
from koopmanlib.dictionary import PsiNN_obs
from tensorflow.keras.optimizers import Adam
import json
import sys
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

data_path = config['data_settings']['data_path']
weights_path = config['nn_settings']['weights_path']
forcing_type = config['data_settings']['forcing_type']

n_psi_train = config['nn_settings']['n_psi_train']


Nx = config['data_settings']['Nx']
L = 2 * np.pi
dx = L / (Nx - 1.0)
target_dim = Nx
param_dim = 3

n_psi = 1 + 1 + 1 + n_psi_train

dict_layer_size = config['nn_settings']['dict_layer_size']
K_layer_size = config['nn_settings']['K_layer_size']

pknn_epochs = config['nn_settings']['pknn_epochs']

# Load data
dict_data = np.load(os.path.join(data_path, 'data_kdv_' +
                    forcing_type+'.npy'), allow_pickle=True)

data_x = dict_data[()]['data_x']
data_y = dict_data[()]['data_y']
data_u = dict_data[()]['data_u']

# PK-NN
dic_pk = PsiNN_obs(layer_sizes=dict_layer_size, n_psi_train=n_psi_train, dx=dx)

from koopmanlib.K_structure import Model_K_u_Layer

model_K_u = Model_K_u_Layer(layer_sizes=K_layer_size, 
                                n_psi=n_psi)

solver_pk = KoopmanParametricDLSolver(
    target_dim=target_dim, 
    param_dim=param_dim, 
    n_psi=n_psi, 
    dic=dic_pk, 
    model_K_u=model_K_u)

model_pk, model_K_u_pred_pk = solver_pk.generate_model()


model_pk.summary()

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

history = model_pk.fit(x=[data_x, data_y, data_u],
                       y=zeros_data_y_train,
                       epochs=pknn_epochs,
                       batch_size=200,
                       callbacks=lr_callbacks,
                       verbose=1)


model_pk.save_weights(os.path.join(
    weights_path, 'K_layer/pk_kdv_weights_klayer'+str(K_layer_size[-1])+'.h5'))
