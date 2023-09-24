import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_pde(x_axis, y_axis, data, Nx):
    X, Y = np.meshgrid(x_axis, y_axis)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    c1 = ax1.pcolor(X,Y,data[:,:Nx])
    fig.colorbar(c1, ax=ax1)
    ax1.set_xlabel('space')
    ax1.set_ylabel('time')
    ax1.set_title('$v$')

    c2 = ax2.pcolor(X,Y,data[:,Nx:])
    fig.colorbar(c2, ax=ax2)
    ax2.set_xlabel('space')
    ax2.set_ylabel('time')
    ax2.set_title('$w$')

def plot_pde_comparison(x_axis, y_axis, data_true, data_pred, Nx):
    X, Y = np.meshgrid(x_axis, y_axis)
    fig, axs = plt.subplots(2,2, figsize=(16,12))
    ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    c1 = ax1.pcolormesh(X,Y,data_true[:,:Nx])
    fig.colorbar(c1, ax=ax1)
    ax1.set_xlabel('space')
    ax1.set_ylabel('time')
    # ax1.set_ylim(bottom=0, top=20)
    ax1.set_title('$v$ (true)')

    c2 = ax2.pcolormesh(X,Y,data_pred[:,:Nx])
    fig.colorbar(c2, ax=ax2)
    ax2.set_xlabel('space')
    ax2.set_ylabel('time')
    ax2.set_title('$v$ (prediction)')

    c3 = ax3.pcolormesh(X,Y,data_true[:,Nx:])
    fig.colorbar(c3, ax=ax3)
    ax3.set_xlabel('space')
    ax3.set_ylabel('time')
    ax3.set_title('$w$ (true)')


    c4 = ax4.pcolormesh(X,Y,data_pred[:,Nx:])
    fig.colorbar(c4, ax=ax4)
    ax4.set_xlabel('space')
    ax4.set_ylabel('time')
    ax4.set_title('$w$ (prediction)')

def compute_diff_ratio_one_traj(data_true_list, data_pred_list):

    diff = data_true_list - data_pred_list
    diff_norm = np.square(np.linalg.norm(diff, axis=-1))
    data_true_norm = np.square(np.linalg.norm(data_true_list, axis=-1))

    sum_diff_norm_list = []
    sum_data_true_norm_list = []
    for i in range(diff.shape[0]):
        sum_diff_norm = np.sqrt(np.sum(diff_norm[:i+1]))
        sum_data_true_norm = np.sqrt(np.sum(data_true_norm[:i+1]))
        sum_diff_norm_list.append(sum_diff_norm)
        sum_data_true_norm_list.append(sum_data_true_norm)

    sum_diff_norm_list = np.asarray(sum_diff_norm_list)
    sum_data_true_norm_list = np.asarray(sum_data_true_norm_list)

    ratio = sum_diff_norm_list / sum_data_true_norm_list
    return ratio

def recover_prediction(target, scaler, solver):
    z_curr_ori, data_u, z_next_ori = target.generate_data()

    z_curr_normalized = scaler.transform(z_curr_ori)
    z_next_normalized = scaler.transform(z_next_ori)

    z_init_normalized = tf.reshape(z_curr_normalized[0], shape=(1,-1))
    data_true = z_curr_normalized

    data_pred_list = solver.compute_data_list(traj_len=target.traj_len, 
                                             data_x_init=z_init_normalized, 
                                             data_u=data_u)

    data_pred = data_pred_list[:-1,:]
    return data_true, data_pred, z_curr_ori


def compute_stat_info(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    mean_plus_std = data_mean + data_std
    mean_minus_std = data_mean - data_std
    return data_mean, data_std, mean_plus_std, mean_minus_std
