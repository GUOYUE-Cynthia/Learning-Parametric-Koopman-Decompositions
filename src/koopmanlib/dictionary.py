import numpy as np
import scipy
import tensorflow as tf
from scipy.cluster.vq import kmeans
from tensorflow.keras.layers import Concatenate, Dense, Layer
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx("float64")


class AbstractDictionary:
    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]    
        
        if self.add_constant:
            self.basis_func_number = self.n_psi_train + target_dim + 1
            # Form B matrix
            self.B = np.zeros((self.basis_func_number, target_dim))
            for i in range(0, target_dim):
                self.B[i + 1][i] = 1

        else:
            self.basis_func_number = self.n_psi_train + target_dim
            # Form B matrix
            self.B = np.zeros((self.basis_func_number, target_dim))
            for i in range(0, target_dim):
                self.B[i][i] = 1

        return self.B


class DicNN(Layer):
    """Trainable dictionaries."""

    def __init__(self, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.input_layer = Dense(self.layer_sizes[0], use_bias=False)
        self.hidden_layers = [Dense(s, activation="tanh") for s in layer_sizes]
        self.output_layer = Dense(n_psi_train)
        self.n_psi_train = n_psi_train

    def call(self, inputs):
        psi_x_train = self.input_layer(inputs)
        for layer in self.hidden_layers:
            psi_x_train = psi_x_train + layer(psi_x_train)
        outputs = self.output_layer(psi_x_train)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"layer_sizes": self.layer_sizes, "n_psi_train": self.n_psi_train})
        return config


# class PsiNN(Layer, AbstractDictionary):
#     """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]"""

#     def __init__(self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
#         super().__init__(**kwargs)
#         self.layer_sizes = layer_sizes
#         self.dic_trainable = dic_trainable
#         self.n_psi_train = n_psi_train
#         self.dicNN = self.dic_trainable(
#             layer_sizes=self.layer_sizes, n_psi_train=self.nn_psi_train, name="DicNN"
#         )

#     def call(self, inputs):
#         constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))
#         psi_x_train = self.dicNN(inputs)
#         outputs = Concatenate()([constant, inputs, psi_x_train])
#         return outputs

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "dic_trainable": self.dic_trainable,
#                 "layer_sizes": self.layer_sizes,
#                 "n_psi_train": self.n_psi_train,
#             }
#         )
#         return config
    
class PsiNN(Layer, AbstractDictionary):
    """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]"""


    def __init__(self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, add_constant=True, **kwargs):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_psi_train = n_psi_train
        self.add_constant = add_constant
        if self.n_psi_train != 0:  # Initialize trainable part when n_psi_train is not 0
            self.dicNN = self.dic_trainable(
                layer_sizes=self.layer_sizes, n_psi_train=self.n_psi_train, name="DicNN"
            )

    def call(self, inputs):
        outputs = []

        if self.add_constant:
            constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))
            outputs.append(constant)

        outputs.append(inputs)

        if self.n_psi_train != 0:
            psi_x_train = self.dicNN(inputs)
            outputs.append(psi_x_train)

        if len(outputs) > 1:
            outputs = Concatenate()(outputs)
        else:
            outputs = outputs[0]

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "dic_trainable": self.dic_trainable,
            "layer_sizes": self.layer_sizes,
            "n_psi_train": self.n_psi_train,
            "add_constant": self.add_constant
        })
        return config


class DicRBF(AbstractDictionary, Layer):
    """
    RBF based on notations in
    (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(self, rbf_number=100, regularizer=1e-4):
        self.n_psi_train = rbf_number
        self.regularizer = regularizer
        self.add_constant = True

    def build(self, data):
        self.centers, residual = kmeans(data, self.n_psi_train)

    def call(self, data):
        rbfs = []
        for n in range(self.centers.shape[0]):
            r = scipy.spatial.distance.cdist(data, np.matrix(self.centers[n, :]))
            rbf = scipy.special.xlogy(r**2, r + self.regularizer)
            rbfs.append(rbf)

        rbfs = tf.transpose(tf.squeeze(rbfs))
        rbfs = tf.reshape(rbfs, shape=(data.shape[0], -1))

        ones = tf.ones(shape=(rbfs.shape[0], 1), dtype="float64")
        results = tf.concat([ones, data, rbfs], axis=-1)
        return results


class DicGaussianRBF(AbstractDictionary, Layer):
    """
    RBF based on notations in
    (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(self, rbf_number=100, s=5, regularizer=1e-4):
        self.n_psi_train = rbf_number
        self.regularizer = regularizer
        self.s = s
        self.add_constant = True

    def build(self, data):
        self.centers, residual = kmeans(data, self.n_psi_train)

    def call(self, data):
        rbfs = []
        for n in range(self.centers.shape[0]):
            r = tf.reshape(
                tf.square(tf.norm(data - np.matrix(self.centers[n, :]), axis=-1)), shape=(-1, 1)
            )
            rbf = tf.exp(-self.s * r)
            rbfs.append(rbf)

        rbfs = tf.concat(rbfs, axis=-1)

        ones = tf.ones(shape=(tf.shape(data)[0], 1), dtype="float64")
        results = tf.concat([ones, data, rbfs], axis=-1)
        return results


class DicGaussianRBF_NO_Constant(AbstractDictionary, Layer):
    """
    RBF based on notations in
    (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(self, rbf_number=100, s=5, regularizer=1e-4):
        self.n_psi_train = rbf_number
        self.regularizer = regularizer
        self.s = s

    def build(self, data):
        self.centers, residual = kmeans(data, self.n_psi_train)

    def call(self, data):
        rbfs = []
        for n in range(self.centers.shape[0]):
            r = tf.reshape(
                tf.square(tf.norm(data - np.matrix(self.centers[n, :]), axis=-1)), shape=(-1, 1)
            )
            rbf = tf.exp(-self.s * r)
            rbfs.append(rbf)

        rbfs = tf.concat(rbfs, axis=-1)

        results = tf.concat([data, rbfs], axis=-1)
        return results

    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + target_dim
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, target_dim))
        for i in range(0, target_dim):
            self.B[i][i] = 1
        return self.B


class PsiNN_NO_Constant(Layer, AbstractDictionary):
    """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]"""

    def __init__(self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_psi_train = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes, n_psi_train=self.n_psi_train
        )

    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + target_dim
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, target_dim))
        for i in range(0, target_dim):
            self.B[i][i] = 1
        return self.B

    def call(self, inputs):
        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([inputs, psi_x_train])
        return outputs

    def get_config(self):
        config = super(PsiNN, self).get_config()
        config.update(
            {
                "dic_trainable": self.dic_trainable,
                "layer_sizes": self.layer_sizes,
                "n_psi_train": self.n_psi_train,
            }
        )
        return config


class PsiNN_obs(Layer, AbstractDictionary):
    def __init__(
        self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, dx=0.05, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_psi_train = n_psi_train
        self.dx = dx
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes, n_psi_train=self.n_psi_train
        )

    def call(self, inputs):
        constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))

        obs_mass = self.dx * tf.reshape(tf.math.reduce_sum(inputs, axis=-1), shape=(-1, 1))
        obs_momentum = self.dx * tf.reshape(
            tf.math.reduce_sum(tf.square(inputs), axis=-1), shape=(-1, 1)
        )

        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, obs_mass, obs_momentum, psi_x_train])
        return outputs

    def generate_B_mass(self, inputs):
        # only observe the mass
        obs_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + obs_dim + 1 + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, obs_dim))
        for i in range(0, obs_dim):
            self.B[i + 1][i] = 1
        return self.B

    def generate_B_momentum(self, inputs):
        # only observe the momentum
        obs_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + obs_dim + 1 + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, obs_dim))
        for i in range(0, obs_dim):
            self.B[i + 2][i] = 1
        return self.B

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dic_trainable": self.dic_trainable,
                "layer_sizes": self.layer_sizes,
                "n_psi_train": self.n_psi_train,
            }
        )
        return config


class PsiNN_mass(Layer, AbstractDictionary):
    def __init__(
        self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, dx=0.05, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_psi_train = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes, n_psi_train=self.n_psi_train
        )
        self.dx = dx

    def call(self, inputs):
        constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))

        obs_mass = self.dx * tf.reshape(tf.math.reduce_sum(inputs, axis=-1), shape=(-1, 1))

        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, obs_mass, psi_x_train])
        return outputs

    def generate_B(self, inputs):
        # only observe the mass
        obs_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + obs_dim + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, obs_dim))
        for i in range(0, obs_dim):
            self.B[i + 1][i] = 1
        return self.B

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dic_trainable": self.dic_trainable,
                "layer_sizes": self.layer_sizes,
                "n_psi_train": self.n_psi_train,
            }
        )
        return config


class PsiNN_momentum(Layer, AbstractDictionary):
    def __init__(
        self, dic_trainable=DicNN, layer_sizes=[64, 64], n_psi_train=22, dx=0.05, **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_psi_train = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes, n_psi_train=self.n_psi_train
        )
        self.dx = dx

    def call(self, inputs):
        constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))

        obs_momentum = self.dx * tf.reshape(
            tf.math.reduce_sum(tf.square(inputs), axis=-1), shape=(-1, 1)
        )

        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, obs_momentum, psi_x_train])
        return outputs

    def generate_B(self, inputs):
        # only observe the mass
        obs_dim = inputs.shape[-1]
        self.basis_func_number = self.n_psi_train + obs_dim + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, obs_dim))
        for i in range(0, obs_dim):
            self.B[i + 1][i] = 1
        return self.B

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dic_trainable": self.dic_trainable,
                "layer_sizes": self.layer_sizes,
                "n_psi_train": self.n_psi_train,
            }
        )
        return config
