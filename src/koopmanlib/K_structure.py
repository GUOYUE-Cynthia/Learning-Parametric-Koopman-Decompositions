import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.regularizers import l1

class Model_K_u_Layer(Layer):
    """Trainable K(u)

    """

    def __init__(self, layer_sizes=[64, 64], n_psi=3, **kwargs):
        super(Model_K_u_Layer, self).__init__(**kwargs)
        self.n_psi = n_psi
        self.layer_sizes = layer_sizes
        self.hidden_layers = [Dense(s, activation='tanh') for s in layer_sizes]
        self.output_layer = Dense(self.n_psi**2) # no activation functions
        

    def call(self, inputs):
        hidden_u = inputs
        for layer in self.hidden_layers:
            hidden_u = layer(hidden_u)
        K_u_entry = self.output_layer(hidden_u)
        K_u = tf.reshape(K_u_entry, shape=(-1, self.n_psi, self.n_psi))
        
        return K_u

    def get_config(self):
        config = super(Model_K_u_Layer, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi': self.n_psi
        })
        return config
    
    

class Model_K_u_Layer_One(Layer):
    """Trainable K(u), the first row is (1,0,0,...)

    """

    def __init__(self, layer_sizes=[64, 64], n_psi=3, activation='tanh', **kwargs):
        super(Model_K_u_Layer_One, self).__init__(**kwargs)
        self.n_psi = n_psi
        self.layer_sizes = layer_sizes
        self.hidden_layers = [Dense(s, activation=activation) for s in layer_sizes]
        self.output_layer = Dense(self.n_psi*(n_psi-1), name='K_Dense_Output') # no activation functions
       

    def call(self, inputs):
        hidden_u = inputs
        for layer in self.hidden_layers:
            hidden_u = layer(hidden_u)
        K_u_entry = self.output_layer(hidden_u)
        K_u = tf.reshape(K_u_entry, shape=(-1, self.n_psi, self.n_psi-1))
        Constant_one = tf.constant([[1]] + [[0]] * (self.n_psi-1), dtype='float64')
        Constant_one = tf.tile(Constant_one[tf.newaxis, :, :], multiples=[tf.shape(K_u)[0], 1, 1])
        
        K_u_concat = tf.concat([Constant_one, K_u], axis=2)
        
        return K_u_concat

    def get_config(self):
        config = super(Model_K_u_Layer_One, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi': self.n_psi
        })
        return config