import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

from koopmanlib.dictionary import PsiNN
from koopmanlib.param_solver import KoopmanActuatedDLSolver, KoopmanParametricDLSolver
from koopmanlib.target import ModifiedFHNTarget

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tqdm.keras import TqdmCallback

config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)["fhn_high_dim_u_settings"]

data_path = config["data_settings"]["data_path"]
weights_path = config["nn_settings"]["weights_path"]

Nx = config["data_settings"]["Nx"]
n_traj = config["data_settings"]["n_traj"]
traj_len = config["data_settings"]["traj_len"]


# Load data
data_dict = np.load(
    os.path.join(data_path, "data_high_u_fhn_Nx_" + str(Nx) + ".npy"), allow_pickle=True
)

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
param_dim = 3
n_psi = 1 + target_dim + n_psi_train

dict_layer_size = config["nn_settings"]["dict_layer_size"]
K_layer_size = config["nn_settings"]["K_layer_size"]

pknn_epochs = config["nn_settings"]["pknn_epochs"]
polyK_epochs = config["nn_settings"]["polyK_epochs"]

# Generate fhn equation
x = np.linspace(-10, 10, Nx)


fhn_pde = ModifiedFHNTarget(
    x=x, dt=1e-5, t_step=1e-3, dim=target_dim, param_dim=param_dim, param_input=1e3
)

# Build model


dic_pk = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

from koopmanlib.K_structure import Model_K_u_Layer_One, Model_ResNet_K_u_Layer_One

model_K_u = Model_K_u_Layer_One(layer_sizes=K_layer_size, n_psi=n_psi)

solver_pk = KoopmanParametricDLSolver(
    target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk, model_K_u=model_K_u
)

model_pk, model_K_u_pred_pk = solver_pk.generate_model()

model_pk.summary()

model_pk.compile(optimizer=Adam(0.001), loss="mse")

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.8,
    patience=150,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-7,
)

# # Define the early stopping criteria
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                 min_delta=1e-12,
#                                                 patience=100,
#                                                 verbose=1,
#                                                 mode='auto')

tqdm_callback = TqdmCallback(verbose=1)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(
        weights_path, "norm_high_u_psi_" + str(n_psi_train) + "_model_pk_fhn_Nx_" + str(Nx) + ".h5"
    ),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    save_freq="epoch",
)

zeros_data_z_next_train = tf.zeros_like(dic_pk(data_z_next))

history_pk = model_pk.fit(
    x=[z_curr_normalized, z_next_normalized, data_u],
    y=zeros_data_z_next_train,
    epochs=pknn_epochs,
    batch_size=pknn_epochs,
    validation_split=0.2,
    callbacks=[lr_callback, checkpoint_callback, tqdm_callback],
    verbose=0,
)

training_loss = history_pk.history['loss']
validation_loss = history_pk.history['val_loss']
best_epoch = validation_loss.index(min(validation_loss))
best_loss_pk = training_loss[best_epoch]
best_val_loss_pk = validation_loss[best_epoch]


# Build Dl + Polynomial K
dic_dl_polyK = PsiNN(layer_sizes=dict_layer_size, n_psi_train=n_psi_train)

solver_dl_polyK = KoopmanActuatedDLSolver(dic=dic_dl_polyK,
                                          target_dim=target_dim,
                                          param_dim=param_dim,
                                          n_psi=n_psi,
                                          basis_u_func=fhn_pde.basis_u_func)

model_dl_polyK = solver_dl_polyK.build_model()

solver_dl_polyK.opt_nn_model(data_x=z_curr_normalized,
                             data_u=data_u,
                             data_y=z_next_normalized,
                             zeros_data_y_train=zeros_data_z_next_train,
                             epochs=polyK_epochs,
                             batch_size=200,
                             lr=0.0001,
                             lr_patience=20,
                             lr_decay_factor=0.8,
                             es_patience=50,
                             es_min_delta=1e-8,
                             filepath=os.path.join(
                                weights_path, 'norm_high_u_psi_'+str(n_psi_train)+'_model_dl_polyK_fhn_Nx_'+str(Nx)+'.h5'))



loss_dict = {
    "loss_pk": best_loss_pk,
    "val_loss_pk": best_val_loss_pk,
    "loss_dl_polyK": solver_dl_polyK.loss_best_model,
    "val_loss_dl_polyK": solver_dl_polyK.val_loss_best_model
}

import pandas as pd

# Convert the dictionary to a DataFrame
df = pd.DataFrame([loss_dict])

# Save the DataFrame to a CSV file
df.to_csv('loss_values_fhn_high_dim_u.csv', index=False)
