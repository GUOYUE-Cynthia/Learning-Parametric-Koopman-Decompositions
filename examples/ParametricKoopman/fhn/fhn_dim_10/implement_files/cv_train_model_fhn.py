import json
import os
import sys
import csv

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

from koopmanlib.dictionary import PsiNN
from koopmanlib.param_solver import KoopmanActuatedDLSolver, KoopmanParametricDLSolver
from koopmanlib.target import FitzHughNagumoTarget

from sklearn.model_selection import train_test_split, KFold

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm.keras import TqdmCallback

config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)["fhn_settings"]

data_path = config["data_settings"]["data_path"]
weights_path = config["nn_settings"]["weights_path"]

Nx = config["data_settings"]["Nx"]
n_traj = config["data_settings"]["n_traj"]
traj_len = config["data_settings"]["traj_len"]


# Load data
data_dict = np.load(os.path.join(data_path, "data_fhn_Nx_" + str(Nx) + ".npy"), allow_pickle=True)

data_z_curr = data_dict[()]["data_z_curr"]
data_u = data_dict[()]["data_u"]
data_z_next = data_dict[()]["data_z_next"]

# Normalize data
scaler_z = StandardScaler()
scaler_z.fit(data_z_curr)

z_curr_normalized = scaler_z.transform(data_z_curr)
z_next_normalized = scaler_z.transform(data_z_next)

n_psi_train = config["nn_settings"]["n_psi_train"]

target_dim = Nx * 2
param_dim = 1
n_psi = 1 + target_dim + n_psi_train

# dict_layer_size = config["nn_settings"]["dict_layer_size"]
K_layer_size = config["nn_settings"]["K_layer_size"]

pknn_epochs = config["nn_settings"]["pknn_epochs"]
polyK_epochs = config["nn_settings"]["polyK_epochs"]


dict_layer_size_list = [[32], [64], [128], [32,32], [64,64], [128,128]]

# Generate fhn equation
x = np.linspace(-10, 10, Nx)

fhn_pde = FitzHughNagumoTarget(
    x=x, dt=1e-5, t_step=1e-3, dim=target_dim, param_dim=param_dim, param_input=1e3
)

# Build model
n_splits = 10
kf = KFold(n_splits=n_splits)
results = []

for dict_layer_size in dict_layer_size_list:

    losses = []

    for train_index, test_index in kf.split(z_curr_normalized):
        data_z_curr_train, data_z_curr_test = z_curr_normalized[train_index], z_curr_normalized[test_index]
        data_z_next_train, data_z_next_test = z_next_normalized[train_index], z_next_normalized[test_index]
        data_u_train, data_u_test = data_u[train_index], data_u[test_index]

        dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

        from koopmanlib.K_structure import Model_K_u_Layer_One

        model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi)

        solver_pk = KoopmanParametricDLSolver(
            target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
        )

        model_pk, model_K_u_pred_pk = solver_pk.generate_model()


        model_pk.compile(optimizer=Adam(0.001), loss="mse")

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=80,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-12,
        )


        # Define the TqdmCallback for progress bar
        tqdm_callback = TqdmCallback(verbose=1)

        zeros_data_z_next_train = tf.zeros_like(dic_pk(data_z_next_train))

        history_pk = model_pk.fit(
            x=[data_z_curr_train, data_z_next_train, data_u_train],
            y=zeros_data_z_next_train,
            epochs=pknn_epochs,
            validation_split=0.2,
            batch_size=200,
            callbacks=[lr_callback, tqdm_callback],
            verbose=0,
        )

        zeros_data_z_next_test = tf.zeros_like(dic_pk(data_z_next_test))

        loss_pred_test = model_pk.evaluate(x=[data_z_curr_test, data_z_next_test, data_u_test], y=zeros_data_z_next_test)

        losses.append(loss_pred_test)

    mean_loss = np.mean(losses)

    results.append(
        {
            'dict_layer_size': dict_layer_size,
            'K_layer_size': K_layer_size,
            'mean_loss': mean_loss
        }
    )

    print('Finished training for dict_layer_size: ', dict_layer_size)

csv_file = 'kdv_cv_results.csv'
csv_columns = ['dict_layer_size', 'K_layer_size', 'mean_loss']

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")

    