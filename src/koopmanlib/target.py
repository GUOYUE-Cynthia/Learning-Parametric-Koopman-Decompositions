import tensorflow as tf
import numpy as np
from scipy import signal

from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff


class AbstractODETarget(object):
    def __init__(
            self,
            dt=1e-3,
            t_step=0.25,
            dim=2):
        self.dim = dim
        self.dt = dt
        self.t_step = t_step
        self.n_step = int(t_step / dt)

    def generate_init_data(self, n_traj, traj_len, seed=None):
        data_x = []
        if seed is not None:
            np.random.seed(seed)

        x0 = np.random.uniform(
            size=(
                n_traj,
                self.dim),
            low=self.x_min,
            high=self.x_max)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t]))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                n_traj * traj_len, self.dim))
        return np.asarray(data_x)

    def generate_next_data(self, data_x):
        data_y = self.euler(data_x)
        return data_y

    def rhs(self):
        """RHS Function
        :return: The rhs of one specific ODE
        """
        return NotImplementedError

    def euler(self, x):
        """ODE Solver

        :param x: variable
        :type x: vector (float)
        :return: ODE Solution at t_step after iterating the Euler method n_step times
        :rtype: vector with the same shape as the variable x (float)
        """
        for _ in range(self.n_step):
            x = x + self.dt * self.rhs(x)
        return x


class DuffingOscillator(AbstractODETarget):
    """Duffing equation based on the notation in

    (https://en.wikipedia.org/wiki/Duffing_equation)
    """

    def __init__(
            self,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            delta=0.5,
            alpha=1.0,
            beta=-1.0
    ):
        super(
            DuffingOscillator,
            self).__init__(
            dt,
            t_step,
            dim)
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.x_min = -2
        self.x_max = 2

    def rhs(self, x):
        x1 = tf.reshape(x[:, 0], shape=(x.shape[0], 1))
        x2 = tf.reshape(x[:, 1], shape=(x.shape[0], 1))
        f1 = x2
        f2 = -self.delta * x2 - x1 * (self.beta + self.alpha * x1**2)
        return tf.concat([f1, f2], axis=-1)


class VanderPolOscillator(AbstractODETarget):
    """Van der Pol Oscillator based on the notation in

    (https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
    """

    def __init__(
            self,
            dt=1e-3,
            t_step=0.1,
            dim=2,
            alpha=2.0):
        super(
            VanderPolOscillator,
            self).__init__(
            dt,
            t_step,
            dim)
        self.alpha = alpha
        self.x_min = -5
        self.x_max = 5

    def rhs(self, x):
        x1 = tf.reshape(x[:, 0], shape=(x.shape[0], 1))
        x2 = tf.reshape(x[:, 1], shape=(x.shape[0], 1))
        f1 = x2
        f2 = self.alpha * (1.0 - x1**2) * x2 - x1
        return tf.concat([f1, f2], axis=-1)


class AbstractParamODETarget(AbstractODETarget):
    def __init__(self,
                 dt=1e-3,
                 t_step=0.1,
                 dim=2,
                 param_dim=2):
        super(AbstractParamODETarget, self).__init__(
            dt,
            t_step,
            dim)
        self.param_dim = param_dim

    def euler(self, x, param):
        """ODE Solver

        :param x: variable
        :type x: vector (float)
        :return: ODE Solution at t_step after iterating the Euler method n_step times
        :rtype: vector with the same shape as the variable x (float)
        """
        for _ in range(self.n_step):
            x = x + self.dt * self.rhs(x, param)
        return x

    def generate_init_data(self, n_traj, traj_len, seed_x, seed_param):
        data_x = []
        if seed_x is not None:
            np.random.seed(seed_x)
            x0 = np.random.uniform(
                size=(
                    n_traj,
                    self.dim),
                low=self.x_min,
                high=self.x_max)
        else:
            x0 = np.random.uniform(
                size=(
                    n_traj,
                    self.dim),
                low=self.x_min,
                high=self.x_max)

        if seed_param is not None:
            np.random.seed(seed_param)
            param = np.random.uniform(
                size=(
                    n_traj,
                    self.param_dim),
                low=self.param_min,
                high=self.param_max)
        else:
            param = np.random.uniform(
                size=(
                    n_traj,
                    self.param_dim),
                low=self.param_min,
                high=self.param_max)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                n_traj * traj_len, self.dim))

        repeats_constant = traj_len * tf.ones(shape=(n_traj,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)

        return np.asarray(data_x), np.asarray(param)

    def generate_next_data(self, data_x, param):
        data_y = self.euler(data_x, param)
        return data_y


class DuffingParamTarget(AbstractParamODETarget):
    def __init__(self,
                 dt=1e-3,
                 t_step=0.25,
                 dim=2,
                 param_dim=3):
        super(DuffingParamTarget, self).__init__(
            dt,
            t_step,
            dim,
            param_dim)
        self.x_min = -2
        self.x_max = 2
        self.delta_min = 0
        self.delta_max = 1
        self.alpha_min = 0
        self.alpha_max = 2
        self.beta_min = -2
        self.beta_max = 2

    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        delta = tf.reshape(param[:, 0], shape=(param.shape[0], 1))
        alpha = tf.reshape(param[:, 1], shape=(param.shape[0], 1))
        beta = tf.reshape(param[:, 2], shape=(param.shape[0], 1))
        f1 = x2
        f2 = -delta * x2 - x1 * (beta + alpha * x1**2)
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self, n_param, traj_len, n_traj_per_param, seed_x=123, seed_param=[1, 2, 3]):
        data_x = []
        if seed_x is not None:
            np.random.seed(seed_x)
        x0 = np.random.uniform(
            size=(
                n_param * n_traj_per_param,
                self.dim),
            low=self.x_min,
            high=self.x_max)

        if seed_param is not None:
            np.random.seed(seed_param[0])

        delta = np.random.uniform(
            size=(
                n_param,
                1),
            low=self.delta_min,
            high=self.delta_max)

        if seed_param is not None:
            np.random.seed(seed_param[1])

        alpha = np.random.uniform(
            size=(
                n_param,
                1),
            low=self.alpha_min,
            high=self.alpha_max)

        if seed_param is not None:
            np.random.seed(seed_param[2])

        beta = np.random.uniform(
            size=(
                n_param,
                1),
            low=self.beta_min,
            high=self.beta_max)
        param = np.concatenate((delta, alpha, beta), axis=-1)
        repeats_constant_extend_n_traj = n_traj_per_param * \
            tf.ones(shape=(n_param,), dtype='int32')
        param = tf.repeat(
            param, repeats=repeats_constant_extend_n_traj, axis=0)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                n_param * traj_len * n_traj_per_param, self.dim))

        repeats_constant_extend_trajlen = traj_len * \
            tf.ones(shape=(n_param * n_traj_per_param,), dtype='int32')
        param = tf.repeat(
            param, repeats=repeats_constant_extend_trajlen, axis=0)

        return np.asarray(data_x), np.asarray(param)

    def generate_fix_param_init_data(self, traj_len, n_traj_per_param, fixed_x0, fixed_param):
        data_x = []

        x0 = fixed_x0.reshape(n_traj_per_param, self.dim)

        delta = fixed_param[0] * np.ones(shape=(1, 1))
        alpha = fixed_param[1] * np.ones(shape=(1, 1))
        beta = fixed_param[2] * np.ones(shape=(1, 1))

        param = np.concatenate((delta, alpha, beta), axis=-1)
        repeats_constant_extend_n_traj = n_traj_per_param * \
            tf.ones(shape=(1,), dtype='int32')
        param = tf.repeat(
            param, repeats=repeats_constant_extend_n_traj, axis=0)

        data_x.append(x0)
        for t in range(traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                traj_len, self.dim))

        repeats_constant_extend_traj_len = traj_len * \
            tf.ones(shape=(n_traj_per_param,), dtype='int32')
        param = tf.repeat(
            param, repeats=repeats_constant_extend_traj_len, axis=0)

        return np.asarray(data_x), np.asarray(param)


class VanderPolMathieuTarget(AbstractParamODETarget):
    def __init__(self,
                 mu,
                 dt=1e-3,
                 t_step=1e-2,
                 dim=2,
                 param_dim=1,
                 k1=2,
                 k2=2,
                 k3=2,
                 k4=1,
                 w0=1):
        super(VanderPolMathieuTarget, self).__init__(
            dt,
            t_step,
            dim,
            param_dim)
        self.x_min = -1
        self.x_max = 1
        self.u_min = -1
        self.u_max = 1
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.k3 = self.mu
        self.k4 = k4
        self.w0 = w0
        # k3 is mu in the equation shown in the draft
        # k4 is k3 in the equation shown in the draft
        # will modifiy it in this code later

    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        f1 = x2
        f2 = (self.k1 - self.k2 * x1**2) * x2 - (self.w0**2 + 2 *
                                                 self.k3 * param**2 - self.k3) * x1 + self.k4 * param
        # f2 = -1.0*x1 + 2.0*x2 - 2.0*(x1**2)*x2 - self.mu * (param**2) * x1 + 1.0*param
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self, n_traj, traj_len, seed_x, seed_param):
        data_x = []

        if seed_x is not None:
            np.random.seed(seed_x)
        x0 = np.random.uniform(
            size=(
                n_traj,
                self.dim),
            low=self.x_min,
            high=self.x_max)

        if seed_param is not None:
            np.random.seed(seed_param)
        param = np.random.uniform(
            size=(
                traj_len,
                n_traj,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_x.append(x0)
        for t in range(traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            # print('param', param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))

        return np.asarray(data_x), np.asarray(param)


class FitzHughNagumoTarget(AbstractParamODETarget):
    def __init__(self,
                 x,
                 dt=1e-3,
                 t_step=1e-2,
                 dim=3,
                 param_dim=1,
                 delta=4.0,
                 epsilon=0.03,
                 a0=-0.03,
                 a1=2.0,
                 param_input=1e3):
        super(FitzHughNagumoTarget, self).__init__(
            dt,
            t_step,
            dim,
            param_dim)
        self.u_min = -1
        self.u_max = 1
        self.delta = delta
        self.epsilon = epsilon
        self.a0 = a0
        self.a1 = a1
        self.Nx = int(dim / 2)
        self.x = x
        self.x_step = np.diff(self.x, n=1)[0]
        self.param_input = param_input

    def rhs(self, data_z, param):
        v = data_z[:, :self.Nx].reshape(data_z.shape[0], self.Nx)
        w = data_z[:, self.Nx:].reshape(data_z.shape[0], self.Nx)

        v_minus_1 = v[:, 1].reshape(-1, 1)
        v_N_plus_1 = v[:, -2].reshape(-1, 1)
        v_ghost = np.concatenate((v_minus_1, v, v_N_plus_1), axis=-1)
        vxx = np.diff(v_ghost, n=2) / (self.x_step**2)

        w_minus_1 = w[:, 1].reshape(-1, 1)
        w_N_plus_1 = w[:, -2].reshape(-1, 1)
        w_ghost = np.concatenate((w_minus_1, w, w_N_plus_1), axis=-1)
        wxx = np.diff(w_ghost, n=2) / (self.x_step**2)

        input_term = param*(np.exp(-(self.x + 5)**2) +
                            np.exp(-(self.x**2))+np.exp(-(self.x - 5)**2))
        dvdt = vxx + v - v**3 - w + self.param_input * input_term
        dwdt = self.delta * wxx + self.epsilon * (v - self.a1*w - self.a0)

        return np.concatenate((dvdt, dwdt), axis=-1)

    def generate_data(self, n_traj, traj_len, seed_z, seed_param):
        data_z = []

        # Set random z0
        if seed_z is not None:
            np.random.seed(seed_z)

        ab_list = np.random.randint(
            low=1,
            high=20,
            size=(n_traj, 2, 1))

        a_list = ab_list[:, 0, :]
        b_list = ab_list[:, 1, :]
        v_t0 = np.sin(1/10 * np.pi * a_list * self.x + np.pi/2)
        # w_t0 = 1e-1*np.cos(1/10 * np.pi * b_list *self.x)

        w_t0 = np.zeros(shape=v_t0.shape)

        z0 = np.concatenate((v_t0, w_t0), axis=-1)

        # z0 = np.zeros(shape=z0.shape)

        # # Set a specific z0 initial condition
        # v0 = np.sin((np.pi/10) * self.x + np.pi/2).reshape(1,-1)
        # w0 = np.cos((np.pi/10) * self.x).reshape(1,-1)
        # z0 = np.concatenate((v0,w0), axis=-1)
        # z0 = np.repeat(z0, repeats=(n_traj),axis=0)

        if seed_param is not None:
            np.random.seed(seed_param)
        param = np.random.uniform(
            size=(
                traj_len,
                n_traj,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_z.append(z0)
        for t in range(traj_len - 1):
            z_next = self.euler(data_z[-1],
                                param[t])
            # print('z', data_z[-1])
            # print('param', param[t])
            data_z.append(z_next)

        data_z = tf.reshape(tf.convert_to_tensor(data_z), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))

        data_z_curr = data_z[:-n_traj, :]
        data_z_next = data_z[n_traj:, :]
        param_output = param[:-n_traj, :]

        return np.asarray(data_z_curr), np.asarray(param_output), np.asarray(data_z_next)

    def basis_u_func(self, data_u):
        # Obtain [u, u^2, u^3] when u is in dim=1.
        return tf.concat([data_u, data_u**2, data_u**3], axis=-1)


class ModifiedFHNTarget(FitzHughNagumoTarget):
    def __init__(self,
                 x,
                 dt=1e-3,
                 t_step=1e-2,
                 dim=3,
                 param_dim=3,
                 delta=4.0,
                 epsilon=0.03,
                 a0=-0.03,
                 a1=2.0,
                 param_input=1e3):
        super(ModifiedFHNTarget, self).__init__(
            x,
            dt,
            t_step,
            dim,
            param_dim,
            delta,
            epsilon,
            a0,
            a1,
            param_input)

    def rhs(self, data_z, data_u):
        v = data_z[:, :self.Nx].reshape(data_z.shape[0], self.Nx)
        w = data_z[:, self.Nx:].reshape(data_z.shape[0], self.Nx)

        v_minus_1 = v[:, 1].reshape(-1, 1)
        v_N_plus_1 = v[:, -2].reshape(-1, 1)
        v_ghost = np.concatenate((v_minus_1, v, v_N_plus_1), axis=-1)
        vxx = np.diff(v_ghost, n=2) / (self.x_step**2)

        w_minus_1 = w[:, 1].reshape(-1, 1)
        w_N_plus_1 = w[:, -2].reshape(-1, 1)
        w_ghost = np.concatenate((w_minus_1, w, w_N_plus_1), axis=-1)
        wxx = np.diff(w_ghost, n=2) / (self.x_step**2)

        u1 = data_u[:, 0].reshape(-1, 1)
        u2 = data_u[:, 1].reshape(-1, 1)
        u3 = data_u[:, 2].reshape(-1, 1)

        input_term = u1*np.exp(-(self.x + 5)**2) + u2 * \
            np.exp(-(self.x**2)) + u3*np.exp(-(self.x - 5)**2)
        dvdt = vxx + v - v**3 - w + self.param_input * input_term
        dwdt = self.delta * wxx + self.epsilon * (v - self.a1*w - self.a0)

        return np.concatenate((dvdt, dwdt), axis=-1)

    def basis_u_func(self, data_u):
        # Obtain [u1, u2, u3, u1^2, u1u2,..., u1^3, u1^2u2, ..., u3^3] when u is in dim=3.
        # Define the degree of the polynomial features
        # data_u shape: (traj_len, param_dim) = (traj_len, 3)
        # output shape: (traj_len, 19)
        # Usually, n_traj = 1

        u1 = tf.reshape(data_u[:, 0], (-1, 1))
        u2 = tf.reshape(data_u[:, 1], (-1, 1))
        u3 = tf.reshape(data_u[:, 2], (-1, 1))

        # Compute the polynomial features manually
        u1_2, u2_2, u3_2 = u1**2, u2**2, u3**2
        u1_3, u2_3, u3_3 = u1**3, u2**3, u3**3
        u1_u2, u1_u3, u2_u3 = u1*u2, u1*u3, u2*u3
        u1_2u2, u1_2u3, u2_2u1, u2_2u3, u3_2u1, u3_2u2 = u1_2 * \
            u2, u1_2*u3, u2_2*u1, u2_2*u3, u3_2*u1, u3_2*u2
        u1u2u3 = u1*u2*u3
        basis_u_list = [data_u, u1_2, u2_2, u3_2, u1_u2, u1_u3, u2_u3, u1_3,
                        u2_3, u3_3, u1_2u2, u1_2u3, u2_2u1, u2_2u3, u3_2u1, u3_2u2, u1u2u3]
        basis_u = tf.concat(basis_u_list, axis=-1)
        return basis_u


class KortewegDeVriesTarget(AbstractParamODETarget):
    def __init__(self,
                 x,
                 dt=1e-3,
                 t_step=1e-2,
                 dim=128,
                 param_dim=3,
                 forcing_type='sin',
                 v_list=None,
                 L=2*np.pi):
        super(KortewegDeVriesTarget, self).__init__(
            dt,
            t_step,
            dim,
            param_dim)
        self.umin = -1
        self.umax = 1
        self.x = x
        self.x_step = np.diff(self.x, n=1)[0]
        self.v_list = v_list
        self.L = L
        self.forcing_type = forcing_type

    def kdv_ode(self, t, y, param):
        """Differential equations for the KdV equation, discretized in x."""
        # Compute the x derivatives using the pseudo-spectral method.
        yx = psdiff(y, period=self.L)
        yxxx = psdiff(y, period=self.L, order=3)

        # Compute du/dt.
        dydt = -y*yx - yxxx

        param = param.reshape(1, -1)

        if self.forcing_type == 'sin':
            nonlinear_param = np.sin(np.pi * param)
        elif self.forcing_type == 'linear':
            nonlinear_param = param

        param_spatial = nonlinear_param @ self.v_list

        param_spatial = param_spatial.reshape(-1, )

        rhs = dydt + param_spatial

        return rhs

    def kdv_solution(self, y0, t, param):
        """Use odeint to solve the KdV equation on a periodic domain.

        `y0` is initial condition, `t` is the array of time values at which
        the solution is to be computed, and `L` is the length of the periodic
        domain."""

        t_span = [0, t]
        soln = solve_ivp(self.kdv_ode, t_span, y0, t_eval=np.asarray(
            t_span), args=(param, ), method='RK23')
        return soln

    def compute_obs_error(self, dic, compute_kdv_soln_func, compute_obs_func_model, error_func, y0_pred_list, param_pred_list, dx):
        error_mass_list = []
        error_momentum_list = []

        for y0_pred, param_pred in zip(y0_pred_list, param_pred_list):

            # Compute exact solution
            kdv_soln_pred = compute_kdv_soln_func(
                y0_pred, param_list=param_pred)
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
            error_momentum = error_func(
                kdv_momentum_pred, model_kdv_momentum_pred)

            error_mass_list.append(error_mass)
            error_momentum_list.append(error_momentum)

        error_mass_list = np.asarray(error_mass_list)
        error_momentum_list = np.asarray(error_momentum_list)

        return error_mass_list, error_momentum_list

    def generate_y0(self, seed_IC):
        """Profile of the exact solution to the KdV for a single soliton on the real line."""

        # Set the seed
        np.random.seed(seed_IC)

        b = np.random.uniform(size=(3,))
        b0 = b[0] / np.sum(b)
        b1 = b[1] / np.sum(b)
        b2 = b[2] / np.sum(b)

        IC0 = np.exp(-(self.x-np.pi/2)**2)
        IC1 = -np.sin((self.x/2)**2)
        IC2 = np.exp(-(self.x+np.pi/2)**2)

        IC = b0*IC0 + b1*IC1 + b2*IC2
        return IC

    def generate_data(self, n_traj, traj_len, seed_y0, seed_param):
        np.random.seed(seed_y0)
        seed_IC = np.random.randint(0, 100, size=(n_traj,))

        y0_list = []
        for seed in seed_IC:
            y0 = self.generate_y0(seed)
            y0_list.append(y0)
        y0_list = np.asarray(y0_list)

        np.random.seed(seed_param)
        param_list_group = np.random.uniform(low=0, high=1, size=(
            n_traj, traj_len, self.param_dim)) * (self.umax - self.umin) + self.umin

        soln_outer_list = []
        for y0, param_list in zip(y0_list, param_list_group):
            # Calculate inner solution for each y0 and param_list (for one trajectory)
            soln_inner_list = [y0]
            for param in param_list:
                soln = self.kdv_solution(y0, self.t_step, param)
                y0 = soln.y.T[-1]
                soln_inner_list.append(y0)

            soln_inner_list = np.asarray(soln_inner_list)

            soln_outer_list.append(soln_inner_list)

        soln_outer_list = np.asarray(soln_outer_list)

        data_x = soln_outer_list[:, :-1, :].reshape(-1, self.dim)
        data_y = soln_outer_list[:, 1:, :].reshape(-1, self.dim)
        data_u = param_list_group.reshape(-1, self.param_dim)

        return data_x, data_y, data_u
