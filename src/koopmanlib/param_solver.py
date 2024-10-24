import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class EinsumLayer(tf.keras.layers.Layer):
    """Layer wrapping a single tf.einsum operation.

    Usage:
    psi_y = EinsumLayer("ij,ijk->ik")((psi_x, K_u))
    """

    def __init__(self, equation: str):
        super().__init__()
        self.equation = equation

    def call(self, psi_x, K_u):
        return tf.einsum(self.equation, psi_x, K_u)

    def get_config(self):
        return {"equation": self.equation}


class KoopmanParamDLGeneralSolver:
    def __init__(self, target_dim, param_dim, dic, n_psi):
        self.target_dim = target_dim
        self.param_dim = param_dim
        self.dic = dic
        self.dic_func = dic.call
        self.n_psi = n_psi


class KoopmanParametricDLSolver(KoopmanParamDLGeneralSolver):
    def __init__(self, target_dim, param_dim, dic, n_psi, model_K_u):
        super().__init__(
            target_dim,
            param_dim,
            dic,
            n_psi,
        )
        self.model_K_u = model_K_u

    def generate_model(self):
        inputs_u = Input((self.param_dim,))
        inputs_psi_x = Input((self.n_psi,))
        K_u = self.model_K_u(inputs_u)
        psi_y = EinsumLayer("ij,ijk->ik")(inputs_psi_x, K_u)
        self.model_K_u_pred = Model(
            inputs=[inputs_u, inputs_psi_x], outputs=psi_y, name="K_u_pred"
        )

        # Build the model related to Psi
        inputs_x = Input((self.target_dim,))
        inputs_y = Input((self.target_dim,))
        inputs_u = Input((self.param_dim,))

        psi_x = self.dic_func(inputs_x)
        psi_y = self.dic_func(inputs_y)

        psi_next = self.model_K_u_pred([inputs_u, psi_x])

        outputs = psi_next - psi_y
        self.model = Model(inputs=[inputs_x, inputs_y, inputs_u], outputs=outputs)
        return self.model, self.model_K_u_pred

    def compute_data_list(self, traj_len, data_x_init, data_u):
        data_x_init = tf.reshape(data_x_init, shape=(1, -1))
        data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

        B = self.dic.generate_B(data_x_init)
        data_pred_list = [data_x_init]

        for i in range(traj_len - 1):
            psi_x = self.dic.call(data_pred_list[-1])
            psi_y = self.model_K_u_pred([data_u[i], psi_x])
            y_pred = psi_y @ B
            data_pred_list.append(y_pred)

        data_pred_list = np.squeeze(np.array(data_pred_list))
        return data_pred_list
    
    # def compute_data_list(self, traj_len, data_x_init, data_u):
    #     data_x_init = tf.reshape(data_x_init, shape=(1, -1))
    #     data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

    #     B = self.dic.generate_B(data_x_init)
    #     data_pred_list = [data_x_init]
    #     psi_x = self.dic.call(data_pred_list[-1])

    #     for i in range(traj_len - 1):
    #         psi_x = self.model_K_u_pred([data_u[i], psi_x])
    #         x_pred = psi_x @ B
    #         data_pred_list.append(x_pred)

    #     data_pred_list = np.squeeze(np.array(data_pred_list))
    #     return data_pred_list


class KoopmanLinearDLSolver(KoopmanParamDLGeneralSolver):
    def build_model(self):
        """Build model with trainable dictionary.

        The loss function is ||Psi(y) - K Psi(x)||^2 .
        """

        inputs_u = Input((self.param_dim,))
        inputs_psi_x = Input((self.n_psi,))

        Layer_A = Dense(units=inputs_psi_x.shape[-1],
                        use_bias=False, name="Layer_A", 
                        trainable=False)
        Layer_B = Dense(units=inputs_psi_x.shape[-1], 
                        use_bias=False, name="Layer_B", 
                        trainable=False)

        psi_y = Layer_A(inputs_psi_x) + Layer_B(inputs_u)

        self.model_K_u_pred_linear = Model(inputs=[inputs_u, inputs_psi_x], outputs=psi_y, name="K_u_linear_pred")

        inputs_x = Input((self.target_dim,))
        inputs_u = Input((self.param_dim,))
        inputs_y = Input((self.target_dim,))

        psi_x = self.dic_func(inputs_x)
        psi_y = self.dic_func(inputs_y)

        psi_next = self.model_K_u_pred_linear([inputs_u, psi_x])
        
        outputs = psi_next - psi_y
        self.model = Model(inputs=[inputs_x, inputs_u, inputs_y], outputs=outputs)
        return self.model, self.model_K_u_pred_linear

    def compute_AB(self, dic_func, data_x, data_u, data_y):
        psi_x = dic_func(data_x)
        concat_psix_u = tf.concat([psi_x, data_u], axis=-1)
        psi_y = dic_func(data_y)

        # If the number of samples is larger than the dimension of dictionaries,
        # the pseudo-inverse matrix is the left one we want.
        concat_psix_u_inv = tf.linalg.pinv(concat_psix_u)
        self.AB = tf.matmul(concat_psix_u_inv, psi_y)
        return self.AB

    def train_psi(self, model, data_x, data_u, data_y, zeros_data_y_train, epochs, batch_size=200):
        """Train the trainable part of the dictionary.

        :param model: koopman model
        :type model: model
        :param epochs: the number of training epochs before computing K for each inner training
            epoch
        :type epochs: int
        :return: history
        :rtype: history callback object
        """

        lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=200,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-12,
        )

        history = model.fit(
            x=[data_x, data_u, data_y],
            y=zeros_data_y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=lr_callbacks,
            verbose=1,
        )
        return history

    def build(
        self,
        model,
        data_x,
        data_u,
        data_y,
        zeros_data_y_train,
        epochs,
        batch_size,
        lr,
        lr_patience,
        lr_decay_factor,
        lr_min,
        es_patience,
        es_min_delta,
        filepath,
    ):
        """Train Koopman model and calculate the final information, such as eigenfunctions,
        eigenvalues and K. For each outer training epoch, the koopman dictionary is trained by
        several times (inner training epochs), and then compute matrix K. Iterate the outer
        training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param lr_patience: the patience of learning decay
        :type lr_patience: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        :param lr_min: the minimum learning rate
        :type lr_min: float
        :param es_patience: the patience of early stopping
        :type es_patience: int
        :param es_min_delta: the minimum delta of early stopping
        :type es_min_delta: float
        :param filepath: the path to save the best model
        :type filepath: str
        """

        # Compile the Koopman DL model
        opt = Adam(lr)
        self.model = model
        self.model.compile(optimizer=opt, loss="mse")

        # Training Loop
        self.losses = []
        self.val_losses = []
        for i in range(epochs):
            # 1 step for training PsiNN
            self.history = self.train_psi(
                self.model, data_x, data_u, data_y, zeros_data_y_train, epochs=1, batch_size=200
            )

            # One step for computing K
            self.AB = self.compute_AB(self.dic_func, data_x, data_u, data_y)

            self.A = self.AB[: -self.param_dim, :]
            self.B = self.AB[-self.param_dim :, :]

            self.model_K_u_pred_linear.get_layer("Layer_A").weights[0].assign(self.A)
            self.model_K_u_pred_linear.get_layer("Layer_B").weights[0].assign(self.B)

            print("number of the outer loop:", i)

            self.losses.append(self.history.history["loss"][-1])
            self.val_losses.append(self.history.history["val_loss"][-1])

            # Adjust learning rate:
            if len(self.losses) > lr_patience:
                if all(self.losses[i] > self.losses[i - 1] for i in range(-1, -(lr_patience + 1), -1)):
                    print("Error increased. Decay learning rate")
                    if curr_lr > lr_min:
                        curr_lr = lr_decay_factor * self.model.optimizer.lr
                        self.model.optimizer.lr = max(curr_lr, lr_min)

            # Checkpoint for saving the best model
            if len(self.val_losses) > 1:
                if self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.loss_best_model = self.losses[-1]
                    self.val_loss_best_model = self.val_losses[-1]
                    self.model.save_weights(filepath)

            # Early stopping:
            if len(self.val_losses) > es_patience:
                if all(
                    abs(self.val_losses[-1] - self.val_losses[i - 1]) < es_min_delta
                    for i in range(-1, -(es_patience + 1), -1)
                ):
                    print("Error increased over patience. Stop training")
                    break

    def compute_data_list(self, traj_len, data_x_init, data_u):
        data_x_init = tf.reshape(data_x_init, shape=(1, -1))
        data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

        B = self.dic.generate_B(data_x_init)
        data_pred_list = [data_x_init]

        for i in range(traj_len - 1):
            psi_x = self.dic.call(data_pred_list[-1])
            psi_y = self.model_K_u_pred_linear.get_layer("Layer_A")(psi_x) + self.model_K_u_pred_linear.get_layer("Layer_B")(
                data_u[i]
            )
            y_pred = psi_y @ B
            data_pred_list.append(y_pred)

        data_pred_list = np.squeeze(np.array(data_pred_list))
        return data_pred_list

    # def compute_data_list(self, traj_len, data_x_init, data_u):
    #     data_x_init = tf.reshape(data_x_init, shape=(1, -1))
    #     data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

    #     B = self.dic.generate_B(data_x_init)
    #     data_pred_list = [data_x_init]
    #     psi_x = self.dic.call(data_pred_list[-1])

    #     for i in range(traj_len - 1):
            
    #         psi_x = self.model_K_u_pred_linear.get_layer("Layer_A")(psi_x) + self.model_K_u_pred_linear.get_layer("Layer_B")(
    #             data_u[i]
    #         )
    #         x_pred = psi_x @ B
    #         data_pred_list.append(x_pred)

    #     data_pred_list = np.squeeze(np.array(data_pred_list))
    #     return data_pred_list



class KoopmanBilinearDLSolver(KoopmanParamDLGeneralSolver):
    
    def build_model(self):
        """Build model with trainable dictionary.

        The loss function is ||Psi(y) - K Psi(x)||^2 .
        """
        # Build the model related to Psi
        inputs_u = Input((self.param_dim,))
        inputs_psi_x = Input((self.n_psi,))

        # u_psix: Multiply each dimension (scalar) of inputs_u (vector) on psi_x (vector)
        u_psix = tf.einsum("ij,ik->kij", inputs_psi_x, inputs_u)

        u_psix_list = []
        for curr in u_psix:
            u_psix_list.append(curr)

        u_psix_list = tf.concat(u_psix_list, axis=-1)

        Layer_A = Dense(units=inputs_psi_x.shape[-1], 
                        use_bias=False, 
                        name="Layer_A",
                        trainable=False)
        Layer_B = Dense(units=inputs_psi_x.shape[-1], 
                        use_bias=False, name="Layer_B", 
                        trainable=False)

        psi_y = Layer_A(inputs_psi_x) + Layer_B(u_psix_list)

        self.model_K_u_pred_bilinear = Model(
            inputs=[inputs_u, inputs_psi_x], outputs=psi_y, name="K_u_bilinear_pred")

        inputs_x = Input((self.target_dim,))
        inputs_u = Input((self.param_dim,))
        inputs_y = Input((self.target_dim,))

        psi_x = self.dic_func(inputs_x)
        psi_y = self.dic_func(inputs_y)

        psi_next = self.model_K_u_pred_bilinear([inputs_u, psi_x])

        outputs = psi_next - psi_y
        self.model = Model(inputs=[inputs_x, inputs_u, inputs_y], outputs=outputs)
        return self.model, self.model_K_u_pred_bilinear



    def compute_AB(self, dic_func, data_x, data_u, data_y):
        psi_x = dic_func(data_x)
        u_psix = tf.einsum("ij,ik->kij", psi_x, data_u)
        u_psix_list = []
        for curr in u_psix:
            u_psix_list.append(curr)

        u_psix_list = tf.concat(u_psix_list, axis=-1)

        concat_psix_psix_u = tf.concat([psi_x, u_psix_list], axis=-1)
        psi_y = dic_func(data_y)

        # If the number of samples is larger than the dimension of dictionaries,
        # the pseudo-inverse matrix is the left one we want.
        concat_psix_psix_u_inv = tf.linalg.pinv(concat_psix_psix_u)
        self.AB = tf.matmul(concat_psix_psix_u_inv, psi_y)
        return self.AB

    def train_psi(self, model, data_x, data_u, data_y, zeros_data_y_train, epochs, batch_size=200):
        """Train the trainable part of the dictionary.

        :param model: koopman model
        :type model: model
        :param epochs: the number of training epochs before computing K for each inner training
            epoch
        :type epochs: int
        :return: history
        :rtype: history callback object
        """

        lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=200,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-12,
        )

        history = model.fit(
            x=[data_x, data_u, data_y],
            y=zeros_data_y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=lr_callbacks,
            verbose=1,
        )
        return history

    def build(
        self,
        model,
        data_x,
        data_u,
        data_y,
        zeros_data_y_train,
        epochs,
        batch_size,
        lr,
        lr_patience,
        lr_decay_factor,
        lr_min,
        es_patience,
        es_min_delta,
        filepath,
    ):
        """Train Koopman model and calculate the final information, such as eigenfunctions,
        eigenvalues and K. For each outer training epoch, the koopman dictionary is trained by
        several times (inner training epochs), and then compute matrix K. Iterate the outer
        training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param lr_patience: the patience of learning decay
        :type lr_patience: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        :param lr_min: the minimum learning rate
        :type lr_min: float
        :param es_patience: the patience of early stopping
        :type es_patience: int
        :param es_min_delta: the minimum delta of early stopping
        :type es_min_delta: float
        :param filepath: the path to save the best model
        :type filepath: str
        """

        # Compile the Koopman DL model
        opt = Adam(lr)
        self.model = model
        self.model.compile(optimizer=opt, loss="mse")

        # Training Loop
        self.losses = []
        self.val_losses = []
        for i in range(epochs):
            # 1 step for training PsiNN
            self.history = self.train_psi(
                self.model, data_x, data_u, data_y, zeros_data_y_train, epochs=1, batch_size=200
            )

            # One step for computing K
            self.AB = self.compute_AB(self.dic_func, data_x, data_u, data_y)

            self.A = self.AB[: -self.param_dim * self.n_psi, :]
            self.B = self.AB[-self.param_dim * self.n_psi :, :]

            self.model_K_u_pred_bilinear.get_layer("Layer_A").weights[0].assign(self.A)
            self.model_K_u_pred_bilinear.get_layer("Layer_B").weights[0].assign(self.B)

            print("number of the outer loop:", i)
            # if i % log_interval == 0:
            #     losses.append(self.history.history['loss'][-1])

            #     # Adjust learning rate:
            #     if len(losses) > 2:
            #         if losses[-1] > losses[-2]:
            #             print("Error increased. Decay learning rate")
            #             curr_lr = lr_decay_factor * self.model.optimizer.lr
            #             self.model.optimizer.lr = curr_lr

            self.losses.append(self.history.history["loss"][-1])
            self.val_losses.append(self.history.history["val_loss"][-1])

            # Adjust learning rate:
            if len(self.losses) > lr_patience:
                if all(self.losses[i] > self.losses[i - 1] for i in range(-1, -(lr_patience + 1), -1)):
                    print("Error increased. Decay learning rate")
                    if curr_lr > lr_min:
                        curr_lr = lr_decay_factor * self.model.optimizer.lr
                        self.model.optimizer.lr = max(curr_lr, lr_min)

            # Checkpoint for saving the best model
            if len(self.val_losses) > 1:
                if self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.loss_best_model = self.losses[-1]
                    self.val_loss_best_model = self.val_losses[-1]
                    self.model.save_weights(filepath)

            # Early stopping:
            if len(self.val_losses) > es_patience:
                if all(
                    abs(self.val_losses[-1] - self.val_losses[i - 1]) < es_min_delta
                    for i in range(-1, -(es_patience + 1), -1)
                ):
                    print("Error increased over patience. Stop training")
                    break

    def compute_data_list(self, traj_len, data_x_init, data_u):
        data_x_init = tf.reshape(data_x_init, shape=(1, -1))
        data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

        B = self.dic.generate_B(data_x_init)
        data_pred_list = [data_x_init]

        for i in range(traj_len - 1):
            psi_x = self.dic.call(data_pred_list[-1])
            u_psix = tf.einsum("ij,ik->kij", psi_x, data_u[i])
            u_psix_list = []
            for curr in u_psix:
                u_psix_list.append(curr)
            u_psix_list = tf.concat(u_psix_list, axis=-1)

            psi_y = self.model_K_u_pred_bilinear.get_layer("Layer_A")(psi_x) + self.model_K_u_pred_bilinear.get_layer("Layer_B")(
                u_psix_list
            )
            y_pred = psi_y @ B
            data_pred_list.append(y_pred)

        data_pred_list = np.squeeze(np.array(data_pred_list))
        return data_pred_list

    # def compute_data_list(self, traj_len, data_x_init, data_u):
    #     data_x_init = tf.reshape(data_x_init, shape=(1, -1))
    #     data_u = tf.reshape(data_u, shape=(data_u.shape[0], 1, -1))

    #     B = self.dic.generate_B(data_x_init)
    #     data_pred_list = [data_x_init]
    #     psi_x = self.dic.call(data_pred_list[-1])

    #     for i in range(traj_len - 1):
            
    #         u_psix = tf.einsum("ij,ik->kij", psi_x, data_u[i])
    #         u_psix_list = []
    #         for curr in u_psix:
    #             u_psix_list.append(curr)
    #         u_psix_list = tf.concat(u_psix_list, axis=-1)

    #         psi_x = self.model_K_u_pred_bilinear.get_layer("Layer_A")(psi_x) + self.model_K_u_pred_bilinear.get_layer("Layer_B")(
    #             u_psix_list
    #         )
    #         x_pred = psi_x @ B
    #         data_pred_list.append(x_pred)

    #     data_pred_list = np.squeeze(np.array(data_pred_list))
    #     return data_pred_list


class KoopmanActuatedDLSolver(KoopmanParamDLGeneralSolver):
    def __init__(self, target_dim, param_dim, dic, n_psi, basis_u_func):
        super().__init__(target_dim, param_dim, dic, n_psi)
        self.basis_u_func = basis_u_func
        # Need to set basis_u_func as input in this class
        # because this function will be used to build the basis of K(u).

    def build_model(self):
        """Build model with trainable dictionary.

        The loss function is ||Psi(y) - K Psi(x)||^2 .
        """
        inputs_x = Input((self.target_dim,))
        inputs_u = Input((self.param_dim,))
        inputs_y = Input((self.target_dim,))

        psi_x = self.dic_func(inputs_x)
        # u_psix: Multiply each dimension (scalar) of [u, u^2, u^3] (vector) on psi_x (vector)
        basis_u = self.basis_u_func(inputs_u)
        u_psix = tf.einsum("ij,ik->kij", psi_x, basis_u)
        u_psix_list = []
        for curr in u_psix:
            u_psix_list.append(curr)

        u_psix_list = tf.concat(u_psix_list, axis=-1)
        concat_psix_psix_u = tf.concat([psi_x, u_psix_list], axis=-1)
        # This helps to change the basis functions of u from [u, u^2, u^3] to [1, u, u^2, u^3].

        psi_y = self.dic_func(inputs_y)

        Layer_Ks = Dense(units=psi_y.shape[-1], use_bias=False, name="Layer_Ks", trainable=False)

        psi_next = Layer_Ks(concat_psix_psix_u)

        outputs = psi_next - psi_y
        self.model = Model(inputs=[inputs_x, inputs_u, inputs_y], outputs=outputs)
        return self.model

    def compute_Ks(self, dic_func, data_x, data_u, data_y):
        psi_x = dic_func(data_x)
        basis_u = self.basis_u_func(data_u)
        u_psix = tf.einsum("ij,ik->kij", psi_x, basis_u)
        u_psix_list = []
        for curr in u_psix:
            u_psix_list.append(curr)

        u_psix_list = tf.concat(u_psix_list, axis=-1)
        concat_psix_psix_u = tf.concat([psi_x, u_psix_list], axis=-1)
        psi_y = dic_func(data_y)

        # If the number of samples is larger than the dimension of dictionaries,
        # the pseudo-inverse matrix is the left one we want.
        concat_psix_psix_u_inv = tf.linalg.pinv(concat_psix_psix_u)
        Ks = tf.matmul(concat_psix_psix_u_inv, psi_y)
        return Ks

    def train_psi(self, model, data_x, data_u, data_y, zeros_data_y_train, epochs, batch_size=200):
        """Train the trainable part of the dictionary.

        :param model: koopman model
        :type model: model
        :param epochs: the number of training epochs before computing K for each inner training
            epoch
        :type epochs: int
        :return: history
        :rtype: history callback object
        """

        lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.1,
            patience=200,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-12,
        )

        history = model.fit(
            x=[data_x, data_u, data_y],
            y=zeros_data_y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=lr_callbacks,
            validation_split=0.2,
            verbose=1,
        )
        return history

    def opt_nn_model(
        self,
        data_x,
        data_u,
        data_y,
        zeros_data_y_train,
        epochs,
        batch_size,
        lr,
        lr_patience,
        lr_decay_factor,
        es_patience,
        es_min_delta,
        filepath
    ):
        """Train Koopman model and calculate the final information, such as eigenfunctions,
        eigenvalues and K. For each outer training epoch, the koopman dictionary is trained by
        several times (inner training epochs), and then compute matrix K. Iterate the outer
        training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param lr_patience: the patience of learning decay
        :type lr_patience: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        :param es_patience: the patience of early stopping
        :type es_patience: int
        :param es_min_delta: the minimum delta of early stopping
        :type es_min_delta: float
        """

        # Compile the Koopman DL model
        opt = Adam(lr)
        self.model.compile(optimizer=opt, loss="mse")

        # Training Loop
        self.losses = []
        self.val_losses = []
        for i in range(epochs):
            # One step for computing K
            self.Ks = self.compute_Ks(self.dic_func, data_x, data_u, data_y)

            self.model.get_layer("Layer_Ks").weights[0].assign(self.Ks)

            # Two steps for training PsiNN
            self.history = self.train_psi(
                self.model, data_x, data_u, data_y, zeros_data_y_train, epochs=1, batch_size=200
            )

            print("number of the outer loop:", i)

            self.losses.append(self.history.history["loss"][-1])
            self.val_losses.append(self.history.history["val_loss"][-1])

            # Adjust learning rate:
            if len(self.losses) > lr_patience:
                if all(self.losses[i] > self.losses[i - 1] for i in range(-1, -(lr_patience + 1), -1)):
                    print("Error increased. Decay learning rate")
                    curr_lr = lr_decay_factor * self.model.optimizer.lr
                    self.model.optimizer.lr = curr_lr

            # Checkpoint for saving the best model
            if len(self.val_losses) > 1:
                if self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.loss_best_model = self.losses[-1]
                    self.val_loss_best_model = self.val_losses[-1]
                    self.model.save_weights(filepath)

            # Early stopping:
            if len(self.val_losses) > es_patience:
                if all(
                    abs(self.val_losses[-1] - self.val_losses[i - 1]) < es_min_delta
                    for i in range(-1, -(es_patience + 1), -1)
                ):
                    print("Error increased over patience. Stop training")
                    break

    def opt_rbf_model(self, data_x, data_u, data_y):
        # self.model = model

        # One step for computing K
        self.Ks = self.compute_Ks(self.dic_func, data_x, data_u, data_y)

        self.model.get_layer("Layer_Ks").weights[0].assign(self.Ks)
        return self.model

    # Only test on one trajectory
    def compute_data_list(self, traj_len, data_x_init, data_u):

        data_x_init = tf.reshape(data_x_init, shape=(1, -1))
        basis_u = self.basis_u_func(data_u)

        basis_u = tf.reshape(basis_u, shape=(basis_u.shape[0], 1, -1))

        B = self.dic.generate_B(data_x_init)
        data_pred_list = [data_x_init]

        for i in range(traj_len - 1):
            psi_x = self.dic.call(data_pred_list[-1])
            u_psix = tf.einsum("ij,ik->kij", psi_x, basis_u[i])
            u_psix_list = []
            for curr in u_psix:
                u_psix_list.append(curr)
            u_psix_list = tf.concat(u_psix_list, axis=-1)
            concat_psix_psix_u = tf.concat([psi_x, u_psix_list], axis=-1)

            psi_y = self.model.get_layer("Layer_Ks")(concat_psix_psix_u)
            y_pred = psi_y @ B
            data_pred_list.append(y_pred)

        data_pred_list = np.squeeze(np.asarray(data_pred_list))
        return data_pred_list
    
    #     # Only test on one trajectory
    # def compute_data_list(self, traj_len, data_x_init, data_u):

    #     data_x_init = tf.reshape(data_x_init, shape=(1, -1))
    #     basis_u = self.basis_u_func(data_u)

    #     basis_u = tf.reshape(basis_u, shape=(basis_u.shape[0], 1, -1))

    #     B = self.dic.generate_B(data_x_init)
    #     data_pred_list = [data_x_init]

    #     psi_x = self.dic.call(data_pred_list[-1])

    #     for i in range(traj_len - 1):
    #         u_psix = tf.einsum("ij,ik->kij", psi_x, basis_u[i])
    #         u_psix_list = []
    #         for curr in u_psix:
    #             u_psix_list.append(curr)
    #         u_psix_list = tf.concat(u_psix_list, axis=-1)
    #         concat_psix_psix_u = tf.concat([psi_x, u_psix_list], axis=-1)

    #         psi_x = self.model.get_layer("Layer_Ks")(concat_psix_psix_u)
    #         x_pred = psi_x @ B
    #         data_pred_list.append(x_pred)

    #     data_pred_list = np.squeeze(np.asarray(data_pred_list))
    #     return data_pred_list
