import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

import sys
import json
import os

from koopmanlib.target import VanderPolMathieuTarget

target_dim = 2
param_dim = 1

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

save_path = config['data_settings']['data_path']

n_init = config['data_settings']['n_init']
traj_len = config['data_settings']['traj_len']
mu = config['data_settings']['mu']

k1, k2, k4, w0 = 2,2,1,1

vdp_mathieu = VanderPolMathieuTarget(mu=mu,
                                     n_init=n_init,
                                     traj_len=traj_len,
                                     dim=target_dim,
                                     param_dim=param_dim, 
                                     k1=k1,
                                     k2=k2,
                                     k4=k4,
                                     w0=w0,
                                     seed_x=123, 
                                     seed_param=1)
data_x, data_u = vdp_mathieu.generate_init_data()
data_y = vdp_mathieu.generate_next_data(data_x, data_u)

dict_data = {'data_x': data_x,
              'data_y': data_y,
              'data_u': data_u}

np.save(os.path.join(save_path,'vdpm_data_mu_'+str(mu)+'.npy'), dict_data)


# n_psi_train = config['nn_settings']['n_psi_train']
# n_psi = 1 + target_dim + n_psi_train
# dic_pk = PsiNN(layer_sizes=[16], n_psi_train=n_psi_train)

# solver_pk = KoopmanParametricDLSolver(target_dim=target_dim, param_dim=param_dim, n_psi=n_psi, dic=dic_pk)

# model_pk, model_K_u_pred_pk = solver_pk.generate_model(layer_sizes=[16])

# model_pk.summary()

# zeros_data_y_train = tf.zeros_like(dic_pk(data_y))

# model_pk.compile(optimizer=Adam(0.001),
#              loss='mse')

# lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
#                                                     factor=0.1,
#                                                     patience=200,
#                                                     verbose=0,
#                                                     mode='auto',
#                                                     min_delta=0.0001,
#                                                     cooldown=0,
#                                                     min_lr=1e-12)

# history = model_pk.fit(x=[data_x, data_y, data_u], 
#                     y=zeros_data_y_train, 
#                     epochs=500, 
#                     batch_size=200,
#                     callbacks=lr_callbacks,
#                     verbose=1)

# # plt.plot(history.history['loss'])
# # plt.yscale('log')
# # plt.title('Training Loss (Duffing Param)')

# model_pk.save_weights('results/vdpm/model_pk_vdpm_mu_'+str(mu)+'.h5')




# # Linear Model: Dynamics is $Az +Bu$

# dic_linear = PsiNN(layer_sizes=[16], n_psi_train=n_psi_train)

# solver_linear = KoopmanLinearDLSolver(dic=dic_linear, target_dim=target_dim, param_dim=param_dim, n_psi=n_psi)

# model_linear = solver_linear.build_model()


# solver_linear.build(model_linear,
#                     data_x,
#                     data_u, 
#                     data_y, 
#                     zeros_data_y_train,
#                     epochs=50,
#                     batch_size=200,
#                     lr=0.0001,
#                     log_interval=20,
#                     lr_decay_factor=0.1)

# model_linear.save_weights('results/vdpm/model_linear_vdpm_mu_'+str(mu)+'.h5')



# # Bilinear Model: Dynamics is $Az + \sum_{i=1}^{N_{u}}B_{i}zu_{i}$

# dic_bilinear = PsiNN(layer_sizes=[16], n_psi_train=n_psi_train)

# solver_bilinear = KoopmanBilinearDLSolver(dic=dic_bilinear, target_dim=target_dim, param_dim=param_dim, n_psi=n_psi)

# model_bilinear = solver_bilinear.build_model()

# # model_bilinear.summary()

# # model_bilinear.weights

# solver_bilinear.build(model_bilinear,
#                     data_x,
#                     data_u, 
#                     data_y, 
#                     zeros_data_y_train,
#                     epochs=50,
#                     batch_size=200,
#                     lr=0.0001,
#                     log_interval=20,
#                     lr_decay_factor=0.1)

# model_bilinear.save_weights('results/vdpm/model_bilinear_vdpm_mu_'+str(mu)+'.h5')
