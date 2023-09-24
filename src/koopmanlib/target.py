import tensorflow as tf
import numpy as np
from scipy import signal

from sklearn.preprocessing import PolynomialFeatures


class AbstractODETarget(object):
    def __init__(
            self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None):
        self.n_init = n_init
        self.traj_len = traj_len
        self.n_data = n_init * traj_len
        self.dim = dim
        self.dt = dt
        self.t_step = t_step
        self.n_step = int(t_step / dt)
        self.seed = seed

    def generate_init_data(self):
        data_x = []
        if self.seed is not None:
            np.random.seed(self.seed)
            
        
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t]))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
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
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None,
            delta=0.5,
            alpha=1.0,
            beta=-1.0
            ):
        super(
            DuffingOscillator,
            self).__init__(
            n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
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
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.1,
            dim=2,
            seed=None,
            alpha=2.0):
        super(
            VanderPolOscillator,
            self).__init__(
            n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
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
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.1,
            dim=2,
            param_dim=2,
            seed_x=None,
            seed_param=None):
        super(AbstractParamODETarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim)
        self.param_dim = param_dim
        self.seed_x = seed_x
        self.seed_param = seed_param
        
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

    # def euler(self, x, param):
    #     """ODE Solver

    #     :param x: variable
    #     :type x: vector (float)
    #     :return: ODE Solution at t_step after iterating the Euler method n_step times
    #     :rtype: vector with the same shape as the variable x (float)
    #     """
    #     for _ in range(self.n_step):
    #         k1 = self.rhs(x, param)
    #         k2 = self.rhs(x + 1/2 * self.dt * k1, param)
    #         k3 = self.rhs(x + 1/2 * self.dt * k2, param)
    #         k4 = self.rhs(x + self.dt * k3, param)
    #     return x + self.dt * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
        
    def generate_init_data(self):
        data_x = []
        if self.seed_x is not None:
            np.random.seed(self.seed_x)
            x0 = np.random.uniform(
                size=(
                    self.n_init,
                    self.dim),
                low=self.x_min,
                high=self.x_max)
        else:
            x0 = np.random.uniform(
                size=(
                    self.n_init,
                    self.dim),
                low=self.x_min,
                high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
            param = np.random.uniform(
                size=(
                    self.n_init,
                    self.param_dim),
                low=self.param_min,
                high=self.param_max)
        else:
            param = np.random.uniform(
                size=(
                    self.n_init,
                    self.param_dim),
                low=self.param_min,
                high=self.param_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        return np.asarray(data_x), np.asarray(param)

    def generate_next_data(self, data_x, param):
        data_y = self.euler(data_x, param)
        return data_y
           

class AffineTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            param_dim=2,
            seed_x=None,
            seed_param=None):
        super(AffineTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -5
        self.x_max = 5
        self.param_min = -2
        self.param_max = 2
        
    def rhs(self, data_x, param):
        A = np.diag([-1/3, -1/2])
        B = np.diag([1/3, 1/4])
        return data_x @ A + param @ B
    

class InputAffineTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            param_dim=2,
            seed_x=None,
            seed_param=None):
        super(InputAffineTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -5
        self.x_max = 5
        self.param_min = -2
        self.param_max = 2
    
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))

        f1 = -0.1*x1 + param
        f2 = x1**2 - x2
        return tf.concat([f1, f2], axis=-1)


class DuffingParamTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            n_traj_per_param,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            param_dim=3,
            seed_x=None,
            seed_param=None):
        super(DuffingParamTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -2
        self.x_max = 2
        self.delta_min = 0
        self.delta_max = 1
        self.alpha_min = 0
        self.alpha_max = 2
        self.beta_min = -2
        self.beta_max = 2
        self.n_traj_per_param = n_traj_per_param
        
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        delta = tf.reshape(param[:, 0], shape=(param.shape[0], 1))
        alpha = tf.reshape(param[:, 1], shape=(param.shape[0], 1))
        beta = tf.reshape(param[:, 2], shape=(param.shape[0], 1))
        f1 = x2
        f2 = -delta * x2 - x1 * (beta + alpha * x1**2)
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []
        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init * self.n_traj_per_param,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param[0])

        delta = np.random.uniform(
            size=(
                self.n_init,
                1),
            low=self.delta_min,
            high=self.delta_max)

        if self.seed_param is not None:
            np.random.seed(self.seed_param[1])
        
        alpha = np.random.uniform(
            size=(
                self.n_init,
                1),
            low=self.alpha_min,
            high=self.alpha_max)

        if self.seed_param is not None:
            np.random.seed(self.seed_param[2])

        beta = np.random.uniform(
            size=(
                self.n_init,
                1),
            low=self.beta_min,
            high=self.beta_max)
        param = np.concatenate((delta, alpha, beta), axis=-1)
        repeats_constant_extend_n_traj = self.n_traj_per_param * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant_extend_n_traj, axis=0)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len * self.n_traj_per_param, self.dim))
        
        repeats_constant_extend_trajlen = self.traj_len * tf.ones(shape=(self.n_init * self.n_traj_per_param,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant_extend_trajlen, axis=0)
        
        return np.asarray(data_x), np.asarray(param)

    def generate_fix_param_init_data(self, fixed_x0, fixed_param):
        data_x = []
        
        # if self.seed_x is not None:
        #     np.random.seed(self.seed_x)
        # x0 = np.random.uniform(
        #     size=(
        #         self.n_init * self.n_traj_per_param,
        #         self.dim),
        #     low=self.x_min,
        #     high=self.x_max)  
        
        x0 = fixed_x0.reshape(self.n_init * self.n_traj_per_param, self.dim)
        
        delta = fixed_param[0] * np.ones(shape=(self.n_init,1))
        alpha = fixed_param[1] * np.ones(shape=(self.n_init,1))
        beta = fixed_param[2] * np.ones(shape=(self.n_init,1))

        param = np.concatenate((delta, alpha, beta), axis=-1)
        repeats_constant_extend_n_traj = self.n_traj_per_param * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant_extend_n_traj, axis=0)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant_extend_trajlen = self.traj_len * tf.ones(shape=(self.n_init * self.n_traj_per_param,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant_extend_trajlen, axis=0)
        
        return np.asarray(data_x), np.asarray(param)


class DampedDuffingTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.01,
            dim=2,
            param_dim=1,
            seed_x=None,
            seed_param=None):
        super(DampedDuffingTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -1
        self.x_max = 1
        self.u_min = -1
        self.u_max = 1
        
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        
        f1 = x2
        f2 = -0.5*x2 - x1*(4*x1**2 -1) + 0.5*param
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []
        # if self.seed_x is not None:
        #     np.random.seed(self.seed_x)
        # x0 = np.random.uniform(
        #     size=(
        #         self.n_init,
        #         self.dim),
        #     low=self.x_min,
        #     high=self.x_max)

        R = 1.0
        if self.seed_x is not None:
            np.random.seed(self.seed_x)

        r = R * np.sqrt(np.random.uniform(size=(self.n_init,1), low=0, high=1))
        theta = np.random.uniform(size=(self.n_init,1)) * 2 * np.math.pi 
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta) 
        x0 = np.concatenate([x1,x2], axis=-1)  
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.traj_len,
                self.n_init,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
        return np.asarray(data_x), np.asarray(param)

        # R = 1.0
        # if self.seed_x is not None:
        #     np.random.seed(self.seed_x)

        # r = R * np.sqrt(np.random.uniform(size=(self.n_init,1), low=0, high=1))
        # theta = np.random.uniform(size=(self.n_init,1)) * 2 * np.math.pi 
        # x1 = r * np.cos(theta)
        # x2 = r * np.sin(theta) 
        # x0 = np.concatenate([x1,x2], axis=-1)  
        # if self.seed_x is not None:
        #     np.random.seed(self.seed_x)
        # x0 = np.random.uniform(
        #     size=(
        #         self.n_init,
        #         self.dim),
        #     low=self.x_min,
        #     high=self.x_max)
  
        # if self.seed_param is not None:
        #     np.random.seed(self.seed_param)    
        # param = np.random.uniform(
        #     size=(
        #         self.n_init,
        #         self.param_dim),
        #     low=self.u_min,
        #     high=self.u_max)

        # data_x.append(x0)
        # for t in range(self.traj_len - 1):
        #     data_x.append(self.euler(data_x[t], param))
        # data_x = tf.reshape(
        #     tf.transpose(
        #         tf.convert_to_tensor(data_x), [
        #             1, 0, 2]), shape=(
        #         self.n_init * self.traj_len, self.dim))
        
        # repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        # param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        # return np.asarray(data_x), np.asarray(param)

    def generate_fix_param_init_data(self, fixed_param):
        data_x = []
        R = 1.0
        if self.seed_x is not None:
            np.random.seed(self.seed_x)

        r = R * np.sqrt(np.random.uniform(size=(self.n_init,1), low=0, high=1))
        theta = np.random.uniform(size=(self.n_init,1)) * 2 * np.math.pi 
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta) 
        x0 = np.concatenate([x1,x2], axis=-1) 
            
        param = fixed_param * np.ones(shape=(self.n_init,1))

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        return np.asarray(data_x), np.asarray(param)


class VanderPolParamTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.1,
            dim=2,
            param_dim=1,
            seed_x=None,
            seed_param=None):
        super(VanderPolParamTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -3
        self.x_max = 3
        self.alpha_min = 1
        self.alpha_max = 4
          
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        f1 = x2
        f2 = param * (1.0 - x1**2) * x2 - x1
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []
        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.n_init,
                self.param_dim),
            low=self.alpha_min,
            high=self.alpha_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        return np.asarray(data_x), np.asarray(param)

    def generate_fix_param_init_data(self, fixed_param):
        data_x = []
        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        param = fixed_param * np.ones(shape=(self.n_init,1))

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        return np.asarray(data_x), np.asarray(param)

    
class VanderPolForcingTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=1e-2,
            dim=2,
            param_dim=1,
            seed_x=None,
            seed_param=None):
        super(VanderPolForcingTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -1
        self.x_max = 1
        self.u_min = -1
        self.u_max = 1
                
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        f1 = 2 * x2
        f2 = -0.8*x1 + 2*x2 - 10*(x1**2)*x2 + param
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []

        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.traj_len,
                self.n_init,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
        return np.asarray(data_x), np.asarray(param)

    # def generate_init_data_v1(self):
    #     data_x = []

    #     if self.seed_x is not None:
    #         np.random.seed(self.seed_x)
    #     x0 = np.random.uniform(
    #         size=(
    #             self.n_init,
    #             self.dim),
    #         low=self.x_min,
    #         high=self.x_max)
            
    #     # if self.seed_param is not None:
    #     #     np.random.seed(self.seed_param)
    #     # param = np.random.uniform(
    #     #     size=(
    #     #         self.traj_len,
    #     #         self.n_init,
    #     #         self.param_dim),
    #     #     low=self.u_min,
    #     #     high=self.u_max)

    #     interval = 30
    #     t = np.linspace(0, 1, self.traj_len, endpoint=False)
    #     scale = (self.traj_len / 2) / interval
    #     control = signal.square(2 * np.pi * scale * t, duty=0.5)
    #     repeat_constant = self.n_init * tf.ones(shape=(self.traj_len, ), dtype='int32')
    #     param = tf.repeat(control, repeats=repeat_constant)
    #     param = tf.reshape(param, shape=(self.traj_len, self.n_init, self.param_dim))

    #     data_x.append(x0)
    #     for t in range(self.traj_len - 1):
    #         x_next = self.euler(data_x[-1], param[t])
    #         data_x.append(x_next)

    #     data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
    #     param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
    #     return np.asarray(data_x), np.asarray(param)



class TolueneHydro(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=0.1,
            t_step=1.0,
            dim=3,
            param_dim=5,
            seed_x=None,
            seed_param=None):
        super(TolueneHydro, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -5
        self.x_max = 5
        self.theta_min = 1e-3
        self.theta_max = 2
        
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        x3 = tf.reshape(data_x[:, 2], shape=(data_x.shape[0], 1))

        theta1 = tf.reshape(param[:, 0], shape=(param.shape[0], 1))
        theta2 = tf.reshape(param[:, 1], shape=(param.shape[0], 1))
        theta3 = tf.reshape(param[:, 2], shape=(param.shape[0], 1))
        theta4 = tf.reshape(param[:, 3], shape=(param.shape[0], 1))
        theta5 = tf.reshape(param[:, 4], shape=(param.shape[0], 1))

        denominator = theta4 * x1 + x2 + theta5 * x3
        r1 = theta1 * theta4 * x1 / denominator
        r_minus_1 = theta3 * x2 / denominator
        r2 = theta2*x2 / denominator

        f1 = -r1 + r_minus_1
        f2 = r1 - r_minus_1 -r2
        f3 = r2
        return tf.concat([f1, f2, f3], axis=-1)

    def generate_init_data(self):
        data_x = []
        if self.seed_x is None:
            x0 = tf.constant([1., 0., 0.], shape=(1,3), dtype='float64')
            x0 = tf.repeat(x0, repeats=[self.n_init], axis=0)
            
        # if self.seed_param is not None:
        #     np.random.seed(self.seed_param)
        #     param = np.random.uniform(
        #         size=(
        #             self.n_init,
        #             5),
        #         low=self.theta_min,
        #         high=self.theta_max)
        # else:
        #     param = np.random.uniform(
        #         size=(
        #             self.n_init,
        #             5),
        #         low=self.theta_min,
        #         high=self.theta_max)

        if self.seed_param is not None:
            np.random.seed(self.seed_param[0])
            theta1 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-1)

            np.random.seed(self.seed_param[1])
            theta2 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-2)

            np.random.seed(self.seed_param[2])
            theta3 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-2)

            np.random.seed(self.seed_param[3])
            theta4 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=0,
                high=2)

            np.random.seed(self.seed_param[4])
            theta5 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=0,
                high=2)
            param = np.concatenate((theta1, theta2, theta3, theta4, theta5), axis=-1)
        else:
            theta1 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-1)

            theta2 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-2)

            theta3 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=1e-3,
                high=1e-2)

            theta4 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=0,
                high=2)

            theta5 = np.random.uniform(
                size=(
                    self.n_init,
                    1),
                low=0,
                high=2)
            param = np.concatenate((theta1, theta2, theta3, theta4, theta5), axis=-1)


        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t], param))

        # data_x shape: (traj_len, n_init, x_dim)
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        
        repeats_constant = self.traj_len * tf.ones(shape=(self.n_init,), dtype='int32')
        param = tf.repeat(param, repeats=repeats_constant, axis=0)
        
        return np.asarray(data_x), np.asarray(param)


class VanderPolForcingFixed(AbstractODETarget):
    """Van der Pol Oscillator based on the notation in

    (https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
    """

    def __init__(
            self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.1,
            dim=2,
            seed=None,
            alpha=2.0):
        super(
            VanderPolForcingFixed,
            self).__init__(
            n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
        self.alpha = alpha
        self.x_min = -5
        self.x_max = 5

    def rhs(self, data_x):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        f1 = 2 * x2
        f2 = -0.8*x1 + 2*x2 - 10*(x1**2)*x2
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):  
        data_x = []
        R = 0.05
        if self.seed is not None:
            np.random.seed(self.seed)
            
        r = R * np.sqrt(np.random.uniform(size=(self.n_init,1), low=0, high=1))
        theta = np.random.uniform(size=(self.n_init,1)) * 2 * np.math.pi 
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta) 
        x0 = np.concatenate([x1,x2], axis=-1)  

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t]))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        return np.asarray(data_x)


class DampedDuffingFixedTarget(AbstractODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None):
        super(DampedDuffingFixedTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
        self.u_min = -1
        self.u_max = 1
        
    def rhs(self, data_x):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        
        f1 = x2
        f2 = -0.5*x2 - x1*(4*x1**2 -1)
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []
        R = 1.0
        if self.seed is not None:
            np.random.seed(self.seed)

        r = R * np.sqrt(np.random.uniform(size=(self.n_init,1), low=0, high=1))
        theta = np.random.uniform(size=(self.n_init,1)) * 2 * np.math.pi 
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta) 
        x0 = np.concatenate([x1,x2], axis=-1)  

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t]))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        return np.asarray(data_x)


class BilinearMotorTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=1e-2,
            dim=2,
            param_dim=1,
            mu=32.2293,
            u_order=1,
            seed_x=None,
            seed_param=None):
        super(BilinearMotorTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.x_min = -1
        self.x_max = 1
        self.u_min = -1
        self.u_max = 1
        self.mu = mu
        self.u_order = int(u_order)
                
    def rhs(self, data_x, param):
        x1 = tf.reshape(data_x[:, 0], shape=(data_x.shape[0], 1))
        x2 = tf.reshape(data_x[:, 1], shape=(data_x.shape[0], 1))
        # f1 = -(Ra / La) * x1 - (km / La) * x2 * param + (ua / La)
        # f2 = -(B / J) * x2 + (km / J) * x1 * param - (tau_l / J)

        # f1 = -39.3153 * x1 - 32.2293 * x2 * param + 19.10828025
        # f2 = -1.6599 * x2 + 22.9478 * x1 * param - 3.333333333

        f1 = -39.3153 * x1 - self.mu * x2 * param ** self.u_order + 19.10828025
        f2 = -1.6599 * x2 + 22.9478 * x1 * param - 3.333333333
        # print('x1 shape', x1.shape)
        # print('x2 shape', x2.shape)
        # print('param shape', param.shape)
        # print('f1 shape', f1.shape)
        # print('f2 shape', f2.shape)
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []

        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.traj_len,
                self.n_init,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
        return np.asarray(data_x), np.asarray(param)

    def generate_init_data_v1(self):
        data_x = []

        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        # if self.seed_param is not None:
        #     np.random.seed(self.seed_param)
        # param = np.random.uniform(
        #     size=(
        #         self.traj_len,
        #         self.n_init,
        #         self.param_dim),
        #     low=self.u_min,
        #     high=self.u_max)

        interval = 30
        t = np.linspace(0, 1, self.traj_len, endpoint=False)
        scale = (self.traj_len / 2) / interval
        control = signal.square(2 * np.pi * scale * t, duty=0.5)
        repeat_constant = self.n_init * tf.ones(shape=(self.traj_len, ), dtype='int32')
        param = tf.repeat(control, repeats=repeat_constant)
        param = tf.reshape(param, shape=(self.traj_len, self.n_init, self.param_dim))

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
        return np.asarray(data_x), np.asarray(param)
    

class VanderPolMathieuTarget(AbstractParamODETarget):
    def __init__(self,
            mu,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=1e-2,
            dim=2,
            param_dim=1,
            k1=2,
            k2=2,
            k3=2,
            k4=1,
            w0=1,
            seed_x=None,
            seed_param=None):
        super(VanderPolMathieuTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
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
        f2 = (self.k1 - self.k2 * x1**2) * x2 - (self.w0**2 + 2 * self.k3 * param**2 - self.k3) * x1 + self.k4 * param
        # f2 = -1.0*x1 + 2.0*x2 - 2.0*(x1**2)*x2 - self.mu * (param**2) * x1 + 1.0*param
        return tf.concat([f1, f2], axis=-1)

    def generate_init_data(self):
        data_x = []

        if self.seed_x is not None:
            np.random.seed(self.seed_x)
        x0 = np.random.uniform(
            size=(
                self.n_init,
                self.dim),
            low=self.x_min,
            high=self.x_max)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.traj_len,
                self.n_init,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            x_next = self.euler(data_x[-1], param[t])
            # print('param', param[t])
            data_x.append(x_next)

        data_x = tf.reshape(tf.convert_to_tensor(data_x), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))
        
        return np.asarray(data_x), np.asarray(param)


class FitzHughNagumoTarget(AbstractParamODETarget):
    def __init__(self,
            n_init,
            traj_len,
            x,
            dt=1e-3,
            t_step=1e-2,
            dim=3,
            param_dim=1,
            delta=4.0,
            epsilon=0.03,
            a0=-0.03,
            a1=2.0,
            param_input=1e3,
            seed_z=None,
            seed_x=None,
            seed_param=None):
        super(FitzHughNagumoTarget, self).__init__(n_init,
            traj_len,
            dt,
            t_step,
            dim,
            param_dim,
            seed_x,
            seed_param)
        self.u_min = -1
        self.u_max = 1
        self.delta = delta
        self.epsilon = epsilon
        self.a0 = a0
        self.a1 = a1
        self.Nx = int(dim /2)
        self.x = x
        self.x_step = np.diff(self.x, n=1)[0]
        self.seed_z = seed_z
        self.seed_param = seed_param
        self.param_input = param_input
                
    def rhs(self, data_z, param):
        v = data_z[:, :self.Nx].reshape(data_z.shape[0], self.Nx)
        w = data_z[:, self.Nx:].reshape(data_z.shape[0], self.Nx)

        v_minus_1 = v[:,1].reshape(-1,1)
        v_N_plus_1 = v[:,-2].reshape(-1,1) 
        v_ghost = np.concatenate((v_minus_1, v, v_N_plus_1), axis=-1)
        vxx = np.diff(v_ghost, n=2) / (self.x_step**2)

        w_minus_1 = w[:,1].reshape(-1,1)
        w_N_plus_1 = w[:,-2].reshape(-1,1) 
        w_ghost = np.concatenate((w_minus_1, w, w_N_plus_1), axis=-1)
        wxx = np.diff(w_ghost, n=2) / (self.x_step**2)

        input_term = param*(np.exp(-(self.x + 5)**2)+ np.exp(-(self.x**2))+np.exp(-(self.x - 5)**2))
        dvdt = vxx + v - v**3 - w + self.param_input * input_term
        dwdt = self.delta * wxx + self.epsilon * (v - self.a1*w -self.a0)

        return np.concatenate((dvdt,dwdt), axis=-1)
    
    # def rhs(self, data_z, param):
    #     v = data_z[:, :self.Nx].reshape(data_z.shape[0], self.Nx)
    #     w = data_z[:, self.Nx:].reshape(data_z.shape[0], self.Nx)

    #     v_minus_1 = v[:,1].reshape(-1,1)
    #     v_N_plus_1 = v[:,-2].reshape(-1,1) 
    #     v_ghost = np.concatenate((v_minus_1, v, v_N_plus_1), axis=-1)
    #     vxx = np.diff(v_ghost, n=2) / (self.x_step**2)

    #     w_minus_1 = w[:,1].reshape(-1,1)
    #     w_N_plus_1 = w[:,-2].reshape(-1,1) 
    #     w_ghost = np.concatenate((w_minus_1, w, w_N_plus_1), axis=-1)
    #     wxx = np.diff(w_ghost, n=2) / (self.x_step**2)

        
    #     dvdt = vxx + (1 + (np.pi**2) / 100) * v
    #     dwdt = wxx + (1 + (np.pi**2) / 100) * w

    #     return np.concatenate((dvdt,dwdt), axis=-1)

    def generate_data(self):
        data_z = []

        # Set random z0
        if self.seed_z is not None:
            np.random.seed(self.seed_z)

        ab_list = np.random.randint(
            low=1, 
            high=20,
            size=(self.n_init,2,1))
        
        a_list = ab_list[:,0,:]
        b_list = ab_list[:,1,:]
        v_t0 = np.sin(1/10 * np.pi * a_list *self.x + np.pi/2)
        # w_t0 = 1e-1*np.cos(1/10 * np.pi * b_list *self.x)

        w_t0 = np.zeros(shape=v_t0.shape)

        z0 = np.concatenate((v_t0, w_t0), axis=-1)

        # z0 = np.zeros(shape=z0.shape)


        # # Set a specific z0 initial condition
        # v0 = np.sin((np.pi/10) * self.x + np.pi/2).reshape(1,-1)
        # w0 = np.cos((np.pi/10) * self.x).reshape(1,-1)
        # z0 = np.concatenate((v0,w0), axis=-1)
        # z0 = np.repeat(z0, repeats=(self.n_init),axis=0)
            
        if self.seed_param is not None:
            np.random.seed(self.seed_param)
        param = np.random.uniform(
            size=(
                self.traj_len,
                self.n_init,
                self.param_dim),
            low=self.u_min,
            high=self.u_max)

        data_z.append(z0)
        for t in range(self.traj_len - 1):
            z_next = self.euler(data_z[-1],
                                param[t])
            # print('z', data_z[-1])
            # print('param', param[t])
            data_z.append(z_next)

        data_z = tf.reshape(tf.convert_to_tensor(data_z), (-1, self.dim))
        param = tf.reshape(tf.convert_to_tensor(param), (-1, self.param_dim))

        data_z_curr = data_z[:-self.n_init, :]
        data_z_next = data_z[self.n_init:, :]
        param_output = param[:-self.n_init, :]

        return np.asarray(data_z_curr), np.asarray(param_output), np.asarray(data_z_next)
    
    def basis_u_func(self, data_u):
        # Obtain [u, u^2, u^3] when u is in dim=1.
        return tf.concat([data_u,data_u**2,data_u**3], axis=-1)
    
    
class ModifiedFHNTarget(FitzHughNagumoTarget):
    def __init__(self,
            n_init,
            traj_len,
            x,
            dt=1e-3,
            t_step=1e-2,
            dim=3,
            param_dim=3,
            delta=4.0,
            epsilon=0.03,
            a0=-0.03,
            a1=2.0,
            param_input=1e3,
            seed_z=None,
            seed_x=None,
            seed_param=None):
        super(ModifiedFHNTarget, self).__init__(n_init,
            traj_len,
            x,
            dt,
            t_step,
            dim,
            param_dim,
            delta,
            epsilon,
            a0,
            a1,
            param_input,
            seed_z,
            seed_x,
            seed_param)

    def rhs(self, data_z, data_u):
        v = data_z[:, :self.Nx].reshape(data_z.shape[0], self.Nx)
        w = data_z[:, self.Nx:].reshape(data_z.shape[0], self.Nx)

        v_minus_1 = v[:,1].reshape(-1,1)
        v_N_plus_1 = v[:,-2].reshape(-1,1) 
        v_ghost = np.concatenate((v_minus_1, v, v_N_plus_1), axis=-1)
        vxx = np.diff(v_ghost, n=2) / (self.x_step**2)

        w_minus_1 = w[:,1].reshape(-1,1)
        w_N_plus_1 = w[:,-2].reshape(-1,1) 
        w_ghost = np.concatenate((w_minus_1, w, w_N_plus_1), axis=-1)
        wxx = np.diff(w_ghost, n=2) / (self.x_step**2)

        u1 = data_u[:,0].reshape(-1,1)
        u2 = data_u[:,1].reshape(-1,1)
        u3 = data_u[:,2].reshape(-1,1)

        input_term = u1*np.exp(-(self.x + 5)**2) + u2*np.exp(-(self.x**2)) + u3*np.exp(-(self.x - 5)**2)
        dvdt = vxx + v - v**3 - w + self.param_input * input_term
        dwdt = self.delta * wxx + self.epsilon * (v - self.a1*w -self.a0)

        return np.concatenate((dvdt,dwdt), axis=-1)
    
    def basis_u_func(self, data_u):
        # Obtain [u1, u2, u3, u1^2, u1u2,..., u1^3, u1^2u2, ..., u3^3] when u is in dim=3.
        # Define the degree of the polynomial features
        # data_u shape: (traj_len, param_dim) = (traj_len, 3)
        # output shape: (traj_len, 19)
        # Usually, n_init = 1

        u1 = tf.reshape(data_u[:,0],(-1,1))
        u2 = tf.reshape(data_u[:,1],(-1,1))
        u3 = tf.reshape(data_u[:,2],(-1,1))

        # Compute the polynomial features manually
        u1_2, u2_2, u3_2 = u1**2, u2**2, u3**2
        u1_3, u2_3, u3_3 = u1**3, u2**3, u3**3
        u1_u2, u1_u3, u2_u3 = u1*u2, u1*u3, u2*u3
        u1_2u2, u1_2u3, u2_2u1, u2_2u3, u3_2u1, u3_2u2 = u1_2*u2, u1_2*u3, u2_2*u1, u2_2*u3, u3_2*u1, u3_2*u2
        u1u2u3 = u1*u2*u3
        basis_u_list = [data_u, u1_2, u2_2, u3_2, u1_u2, u1_u3, u2_u3, u1_3, u2_3, u3_3, u1_2u2, u1_2u3, u2_2u1, u2_2u3, u3_2u1, u3_2u2, u1u2u3]
        basis_u = tf.concat(basis_u_list, axis=-1)
        return basis_u
    

