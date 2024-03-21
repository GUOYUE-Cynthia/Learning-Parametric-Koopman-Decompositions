import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from koopmanlib.dictionary import PsiNN_obs
from koopmanlib.param_solver import (
    KoopmanBilinearDLSolver,
    KoopmanLinearDLSolver,
    KoopmanParametricDLSolver,
)

from tqdm.keras import TqdmCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)["sin"]

data_path = config["data_settings"]["data_path"]
weights_path = config["nn_settings"]["weights_path"]
forcing_type = config["data_settings"]["forcing_type"]

n_psi_train = config["nn_settings"]["n_psi_train"]


Nx = config["data_settings"]["Nx"]
L = 2 * np.pi
dx = L / (Nx - 1.0)
target_dim = Nx
param_dim = 3

n_psi = 1 + 1 + 1 + n_psi_train

dict_layer_size = config["nn_settings"]["dict_layer_size"]
# K_layer_size = config["nn_settings"]["K_layer_size"]

linear_epochs = config["nn_settings"]["linear_epochs"]
bilinear_epochs = config["nn_settings"]["bilinear_epochs"]
pknn_epochs = config["nn_settings"]["pknn_epochs"]

# Load data
dict_data = np.load(
    os.path.join(data_path, "data_kdv_" + forcing_type + ".npy"), allow_pickle=True
)

data_x = dict_data[()]["data_x"]
data_y = dict_data[()]["data_y"]
data_u = dict_data[()]["data_u"]

K_layer_size_list = [[64,64], [128,128]]

for K_layer_size in K_layer_size_list:



    # PK-NN
    dic_pk = PsiNN_obs(layer_sizes=dict_layer_size, n_psi_train=n_psi_train, dx=dx)
    from koopmanlib.K_structure import Model_K_u_Layer_One

    model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi, activation="relu")

    solver_pk = KoopmanParametricDLSolver(
        target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
    )

    model_pk, model_K_u_pred_pk = solver_pk.generate_model()

    model_pk.summary()

    zeros_data_y_train = tf.zeros_like(dic_pk(data_y))

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
    checkpoint_path = os.path.join(weights_path, "K_"+str(K_layer_size[-1])+"_pk_kdv_weights_" + forcing_type + ".h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
        )

    # Define the TqdmCallback for progress bar
    tqdm_callback = TqdmCallback(verbose=1)

    callbacks = [lr_callback, checkpoint_callback, tqdm_callback]

    history_pk = model_pk.fit(
        x=[data_x, data_y, data_u],
        y=zeros_data_y_train,
        validation_split=0.2,
            epochs=pknn_epochs,
            batch_size=200,
            callbacks=callbacks,
            verbose=0
        )



