import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff


def kdv_exact(x, seed):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""

    # Set the seed
    np.random.seed(seed)

    b = np.random.uniform(size=(3,))
    b0 = b[0] / np.sum(b)
    b1 = b[1] / np.sum(b)
    b2 = b[2] / np.sum(b)

    IC0 = np.exp(-(x-np.pi/2)**2)
    IC1 = -np.sin((x/2)**2)
    IC2 = np.exp(-(x+np.pi/2)**2)

    IC = b0*IC0 + b1*IC1 + b2*IC2
    return IC


def kdv_ode(t, y, L, param, v_list, x):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    yx = psdiff(y, period=L)
    yxxx = psdiff(y, period=L, order=3)

    # Compute du/dt.
    dydt = -y*yx - yxxx

    param = param.reshape(1, -1)

    sin_param = np.sin(np.pi * param)

    param_spatial = sin_param @ v_list

    param_spatial = param_spatial.reshape(-1, )

    rhs = dydt + param_spatial

    return rhs

# def kdv_ode(t, y, L, param, v_list):
#     """Differential equations for the KdV equation, discretized in x."""
#     # Compute the x derivatives using the pseudo-spectral method.
#     yx = psdiff(y, period=L)
#     yxxx = psdiff(y, period=L, order=3)

#     # Compute du/dt.
#     dydt = -y*yx - yxxx

#     param = param.reshape(1,-1)

#     param_spatial = param @ v_list

#     param_spatial = param_spatial.reshape(-1, )

#     rhs = dydt + param_spatial

#     return rhs


def kdv_solution(y0, t, L, param, v_list):
    """Use odeint to solve the KdV equation on a periodic domain.

    `y0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain."""

    t_span = [0, t]
    soln = solve_ivp(kdv_ode, t_span, y0, t_eval=np.asarray(
        t_span), args=(L, param, v_list), method='RK23')
    return soln


def compute_error(dic, compute_kdv_soln_func, compute_obs_func_model, error_func, y0_pred_list, param_pred_list, dx):
    error_list = []

    for y0_pred, param_pred in zip(y0_pred_list, param_pred_list):

        # Compute exact solution
        kdv_soln_pred = compute_kdv_soln_func(y0_pred, param_list=param_pred)
        kdv_soln_pred = np.asarray(kdv_soln_pred)

        kdv_momentum_pred = dx * \
            tf.reshape(tf.math.reduce_sum(
                tf.square(kdv_soln_pred), axis=-1), shape=(-1, 1))

        # Compute results for Koopman model
        B_mass = dic.generate_B(kdv_momentum_pred)

        model_kdv_mass_pred = compute_obs_func_model(
            y0_pred, param_pred, B_mass)
        model_kdv_mass_pred = model_kdv_mass_pred.reshape((-1, 1))

        error = error_func(kdv_momentum_pred, model_kdv_mass_pred)
        error_list.append(error)

    error_list = np.asarray(error_list)
    return error_list


def compute_obs_error(dic, compute_kdv_soln_func, compute_obs_func_model, error_func, y0_pred_list, param_pred_list, dx):
    error_mass_list = []
    error_momentum_list = []

    for y0_pred, param_pred in zip(y0_pred_list, param_pred_list):

        # Compute exact solution
        kdv_soln_pred = compute_kdv_soln_func(y0_pred, param_list=param_pred)
        kdv_soln_pred = np.asarray(kdv_soln_pred)

        kdv_mass_pred = dx * \
            tf.reshape(tf.math.reduce_sum(
                kdv_soln_pred, axis=-1), shape=(-1, 1))
        kdv_momentum_pred = dx * \
            tf.reshape(tf.math.reduce_sum(
                tf.square(kdv_soln_pred), axis=-1), shape=(-1, 1))

        # Compute results for Koopman model
        B_mass = dic.generate_B_mass(kdv_mass_pred)
        B_momentum = dic.generate_B_momentum(kdv_mass_pred)

        model_kdv_mass_pred = compute_obs_func_model(
            y0_pred, param_pred, B_mass)
        model_kdv_mass_pred = model_kdv_mass_pred.reshape((-1, 1))

        model_kdv_momentum_pred = compute_obs_func_model(
            y0_pred, param_pred, B_momentum)
        model_kdv_momentum_pred = model_kdv_momentum_pred.reshape((-1, 1))

        error_mass = error_func(kdv_mass_pred, model_kdv_mass_pred)
        error_momentum = error_func(kdv_momentum_pred, model_kdv_momentum_pred)

        error_mass_list.append(error_mass)
        error_momentum_list.append(error_momentum)

    error_mass_list = np.asarray(error_mass_list)
    error_momentum_list = np.asarray(error_momentum_list)

    return error_mass_list, error_momentum_list
