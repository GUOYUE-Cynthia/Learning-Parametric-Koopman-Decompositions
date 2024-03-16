import json
import os
import sys

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
K_layer_size_list = config["nn_settings"]["K_layer_size"]

linear_epochs = config["nn_settings"]["linear_epochs"]
bilinear_epochs = config["nn_settings"]["bilinear_epochs"]
pknn_epochs = config["nn_settings"]["pknn_epochs"]

mu_list = config["data_settings"]["mu"]

loss_dict = {}

def load_data_and_train_models(mu, K_layer_size):
    # Load data
    dict_data = np.load(
        os.path.join(data_path, "vdpm_data_mu_" + str(mu) + ".npy"), allow_pickle=True
    )

    data_x = dict_data[()]["data_x"]
    data_y = dict_data[()]["data_y"]
    data_u = dict_data[()]["data_u"]

    print("data_x.shape = ", data_x.shape)

    # PK-NN
    dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)
    from koopmanlib.K_structure import Model_K_u_Layer_One

    model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi, activation="tanh")

    solver_pk = KoopmanParametricDLSolver(
        target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
    )

    model_pk, model_K_u_pred_pk = solver_pk.generate_model()

    zeros_data_y_train = tf.zeros_like(dic_pk(data_y))

    model_pk.compile(optimizer=Adam(0.001), loss="mse")

    model_pk.summary()

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

    # Define the early stopping criteria
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-12, patience=50, verbose=1, mode="auto"
    )

    checkpoint_path = os.path.join(weights_path, "model_pk_vdpm_mu_" + str(mu) + ".h5")
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

    # Add early_stopping to the list of callbacks
    callbacks = [lr_callback, checkpoint_callback, tqdm_callback]

    history_pk = model_pk.fit(
        x=[data_x, data_y, data_u],
        y=zeros_data_y_train,
        validation_split=0.2,
        epochs=pknn_epochs,
        batch_size=200,
        callbacks=callbacks,
        verbose=0,
    )

    training_loss = history_pk.history['loss']
    validation_loss = history_pk.history['val_loss']
    best_epoch = validation_loss.index(min(validation_loss))
    best_loss_pk = training_loss[best_epoch]
    best_val_loss_pk = validation_loss[best_epoch]




    # # Linear Model: Dynamics is $Az +Bu$

    dic_linear = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

    solver_linear = KoopmanLinearDLSolver(
        dic=dic_linear, target_dim=target_dim, param_dim=param_dim, n_psi=n_psi
    )

    model_linear, model_K_u_pred_linear = solver_linear.build_model()

    solver_linear.build(
        model_linear,
        data_x,
        data_u,
        data_y,
        zeros_data_y_train,
        epochs=linear_epochs,
        batch_size=200,
        lr=0.0001,
        lr_patience=100,
        lr_decay_factor=0.1,
        lr_min=1e-10,
        es_patience=50,
        es_min_delta=1e-9,
        filepath=os.path.join(weights_path, "model_linear_vdpm_mu_" + str(mu) + ".h5"),
    )

    best_loss_linear = solver_linear.loss_best_model
    best_val_loss_linear = solver_linear.val_loss_best_model

    # # Bilinear Model: Dynamics is $Az + \sum_{i=1}^{N_{u}}B_{i}zu_{i}$

    dic_bilinear = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

    solver_bilinear = KoopmanBilinearDLSolver(
        dic=dic_bilinear, target_dim=target_dim, param_dim=param_dim, n_psi=n_psi
    )

    model_bilinear, model_K_u_pred_bilinear = solver_bilinear.build_model()

    solver_bilinear.build(
        model_bilinear,
        data_x,
        data_u,
        data_y,
        zeros_data_y_train,
        epochs=bilinear_epochs,
        batch_size=200,
        lr=0.0001,
        lr_patience=100,
        lr_decay_factor=0.1,
        lr_min=1e-10,
        es_patience=50,
        es_min_delta=1e-9,
        filepath=os.path.join(weights_path, "model_bilinear_vdpm_mu_" + str(mu) + ".h5"),
    )

    best_loss_bilinear = solver_bilinear.loss_best_model
    best_val_loss_bilinear = solver_bilinear.val_loss_best_model

    return {'loss_pk': best_loss_pk,
             'val_loss_pk': best_val_loss_pk, 
             'loss_linear': best_loss_linear, 
             'val_loss_linear': best_val_loss_linear, 
             'loss_bilinear': best_loss_bilinear, 
             'val_loss_bilinear': best_val_loss_bilinear}

loss_dict = {}
for mu, K_layer_size in zip(mu_list, K_layer_size_list):
    loss_dict[f'mu_{mu}'] = load_data_and_train_models(mu, K_layer_size)
    print("mu = ", mu, "done")
    print("K_layer_size = ", K_layer_size, "done")



# Prepare the data for the DataFrame
data = []
for mu, losses in loss_dict.items():
    # Here, 'mu' is your key, and 'losses' is the dictionary with your loss values
    row = [mu] + list(losses.values())  # Start the row with 'mu', then add the loss values
    data.append(row)

# Create a DataFrame
# Since we're not extracting 'K_layer_size' from the key (based on your setup), the columns are adjusted accordingly
df = pd.DataFrame(data, columns=['mu', 'loss_pk', 'val_loss_pk', 'loss_linear', 'val_loss_linear', 'loss_bilinear', 'val_loss_bilinear'])

# Save the DataFrame to a CSV file
df.to_csv('mu_losses_vdpm.csv', index=False)

