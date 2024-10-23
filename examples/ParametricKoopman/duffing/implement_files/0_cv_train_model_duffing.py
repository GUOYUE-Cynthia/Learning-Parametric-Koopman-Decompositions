import json
import os
import sys
import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from koopmanlib.dictionary import PsiNN
from koopmanlib.param_solver import KoopmanParametricDLSolver
from koopmanlib.solver import KoopmanDLSolver
from koopmanlib.target import DuffingParamTarget

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm.keras import TqdmCallback

from koopmanlib.K_structure import Model_K_u_Layer, Model_K_u_Layer_One

from sklearn.model_selection import train_test_split, KFold


config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)


data_path = config["data_settings"]["data_path"]

weights_path = config["nn_settings"]["weights_path"]

n_total_data = config["data_settings"]["n_total_data"]

# Number of the trajectories for each choice of parameters
n_traj_per_param_list = config["data_settings"]["n_traj_per_param"]

results = []

index = 0

# for n_traj_per_param in n_traj_per_param_list:
n_traj_per_param = n_traj_per_param_list[index]

# Number of choices of parameters
n_param = int(n_total_data / n_traj_per_param)
traj_len = config["data_settings"]["traj_len"]
# n_psi_train = config["nn_settings"]["n_psi_train"]

target_dim = 2
param_dim = 3


# Train PK-NN
# Load data

dict_data = np.load(
    os.path.join(
        data_path,
        "duffing_data_n_param_"
        + str(n_param)
        + "_len_"
        + str(traj_len)
        + "_n_traj_per_"
        + str(n_traj_per_param)
        + ".npy",
    ),
    allow_pickle=True,
)

data_x = dict_data[()]["data_x"]
data_y = dict_data[()]["data_y"].numpy()
data_u = dict_data[()]["data_u"]

# print('data_x shape = ', data_x.shape)
# print('data_y shape = ', data_y.shape)
# print('data_u shape = ', data_u.shape)

n_splits = 10
kf = KFold(n_splits=n_splits)

dict_layer_size = config["nn_settings"]['dict_layer_size']

K_layer_size = config["nn_settings"]["K_layer_size"]

n_psi_train_list = [1,5,10,15,20,22,25]

for n_psi_train in n_psi_train_list:

    n_psi = 1 + target_dim + n_psi_train

    losses = []

    for train_index, test_index in kf.split(data_x):
        data_x_train, data_x_test = data_x[train_index], data_x[test_index]
        print('data_x_train shape = ', data_x_train.shape)
        print('data_x_test shape = ', data_x_test.shape)
        data_y_train, data_y_test = data_y[train_index], data_y[test_index]
        data_u_train, data_u_test = data_u[train_index], data_u[test_index]

        dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)
        
        pknn_epochs = config["nn_settings"]["pknn_epochs"]

        model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi)

        solver_pk = KoopmanParametricDLSolver(
            target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
        )

        model_pk, model_K_u_pred_pk = solver_pk.generate_model()

        zeros_data_y_train = tf.zeros_like(dic_pk(data_y_train))

        model_pk.compile(optimizer=Adam(0.001), loss="mse")

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=50,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-12,
        )

        # Define the TqdmCallback for progress bar
        tqdm_callback = TqdmCallback(verbose=1)

        history = model_pk.fit(
            x=[data_x_train, data_y_train, data_u_train],
            y=zeros_data_y_train,
            epochs=pknn_epochs,
            batch_size=200,
            validation_split=0.01,
            callbacks=[lr_callback, tqdm_callback],
            verbose=0,
        )

        zeros_data_y_test = tf.zeros_like(dic_pk(data_y_test))

        loss_pred_test = model_pk.evaluate(x=[data_x_test, data_y_test, data_u_test], y=zeros_data_y_test)

        losses.append(loss_pred_test)

    mean_loss = np.mean(losses)
    results.append({
        'n_psi_train': n_psi_train,
        'n_traj_per_param': n_traj_per_param,
        'dict_layer_size': dict_layer_size,
        'K_layer_size': K_layer_size,
        'mean_loss': mean_loss
    })

    print("Finished training for n_psi_train = ", n_psi_train)

csv_file = str(index)+'_cv_duffing_mse_results.csv'
csv_columns = ['n_psi_train', 'n_traj_per_param', 'dict_layer_size', 'K_layer_size', 'mean_loss']

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)
except IOError:
    print("I/O error")


    
