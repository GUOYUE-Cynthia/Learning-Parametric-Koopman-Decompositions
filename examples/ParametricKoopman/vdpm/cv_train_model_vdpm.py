import json
import os
import sys
import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pandas as pd

from koopmanlib.dictionary import PsiNN
from koopmanlib.param_solver import (
    KoopmanBilinearDLSolver,
    KoopmanLinearDLSolver,
    KoopmanParametricDLSolver,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tqdm.keras import TqdmCallback

# config_file = sys.argv[1]
config_file = "/home/guoyue/Learning-Parametric-Koopman-Decompositions/examples/ParametricKoopman/vdpm/config_vdpm.json"
with open(config_file) as f:
    config = json.load(f)

data_path = config["data_settings"]["data_path"]
weights_path = config["nn_settings"]["weights_path"]

n_psi_train = config["nn_settings"]["n_psi_train"]


target_dim = 2
param_dim = 1

n_psi = 1 + target_dim + n_psi_train

dict_layer_size = config["nn_settings"]["dict_layer_size"]
# K_layer_size_list = config["nn_settings"]["K_layer_size"]

linear_epochs = config["nn_settings"]["linear_epochs"]
bilinear_epochs = config["nn_settings"]["bilinear_epochs"]
pknn_epochs = config["nn_settings"]["pknn_epochs"]

mu_list = config["data_settings"]["mu"]

loss_dict = {}

K_layer_size_list = [[32], [64], [128], [32,32], [64,64], [128,128]]

from sklearn.model_selection import train_test_split, KFold

n_splits = 10
kf = KFold(n_splits=n_splits)

def load_data_and_train_models(mu, K_layer_size):

    results = []
    # Load data
    dict_data = np.load(
        os.path.join(data_path, "vdpm_data_mu_" + str(mu) + ".npy"), allow_pickle=True
    )

    data_x = dict_data[()]["data_x"]
    data_y = dict_data[()]["data_y"]
    data_u = dict_data[()]["data_u"]

    losses = []

    for train_index, test_index in kf.split(data_x):
        data_x_train, data_x_test = data_x[train_index], data_x[test_index]
        data_y_train, data_y_test = data_y[train_index], data_y[test_index]
        data_u_train, data_u_test = data_u[train_index], data_u[test_index]

        print('data_x_train shape = ', data_x_train.shape)
        print('data_x_test shape = ', data_x_test.shape)




        # PK-NN
        dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)
        from koopmanlib.K_structure import Model_K_u_Layer_One

        model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi, activation="tanh")

        solver_pk = KoopmanParametricDLSolver(
            target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
        )

        model_pk, model_K_u_pred_pk = solver_pk.generate_model()

        zeros_data_y_train = tf.zeros_like(dic_pk(data_y_train))

        model_pk.compile(optimizer=Adam(0.001), loss="mse")

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=100,
            verbose=0,
            mode="auto",
            min_delta=1e-4,
            cooldown=0,
            min_lr=1e-10,
        )

        # Define the TqdmCallback for progress bar
        tqdm_callback = TqdmCallback(verbose=1)

        # Add early_stopping to the list of callbacks
        callbacks = [lr_callback, tqdm_callback]

        history_pk = model_pk.fit(
            x=[data_x_train, data_y_train, data_u_train],
            y=zeros_data_y_train,
            validation_split=0.2,
            epochs=pknn_epochs,
            batch_size=200,
            callbacks=callbacks,
            verbose=0,
        )
        zeros_data_y_test = tf.zeros_like(dic_pk(data_y_test))

        loss_pred_test = model_pk.evaluate(x=[data_x_test, data_y_test, data_u_test], y=zeros_data_y_test)

        losses.append(loss_pred_test)

    mean_loss = np.mean(losses)

    return mean_loss
    


for mu in mu_list:
    results = []
    for K_layer_size in K_layer_size_list:
        mean_loss = load_data_and_train_models(mu, K_layer_size)
        results.append({
        'mu': mu,
        'dict_layer_size': dict_layer_size,
        'K_layer_size': K_layer_size,
        'mean_loss': mean_loss
        }) 

    print("Finished training for mu = ", mu)

    csv_file = 'mu_'+str(mu)+'_cv_duffing_mse_results.csv'
    csv_columns = ['mu', 'dict_layer_size', 'K_layer_size', 'mean_loss']
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in results:
                writer.writerow(data)
    except IOError:
        print("I/O error")


