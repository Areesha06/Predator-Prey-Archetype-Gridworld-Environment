# src/baselines/DQN/q_network.py
"""
Configurable feedforward Q-network, built with TensorFlow / Keras.

You control the SHAPE with:
    input_dim = size of the state vector going IN
    output_dim = one Q-value per possible action, coming OUT
    hidden_layers = e.g. [64, 64] = two hidden layers, 64 neurons each.
                     [128, 32, 16] = three layers.

TensorFlow's GradientTape does backpropagation automatically -- we
describe the network and the loss, and TF computes the gradients for us.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# class QNetwork:
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_layers: list,
#         output_dim: int,
#         activation: str = "relu",
#         learning_rate: float = 0.001,
#         grad_clip: float = 10.0,
#         seed: int = None,
#     ):
#         if input_dim <= 0:
#             raise ValueError(f"input_dim must be positive, got {input_dim}")
#         if output_dim <= 0:
#             raise ValueError(f"output_dim must be positive, got {output_dim}")
#         if not hidden_layers or any(h <= 0 for h in hidden_layers):
#             raise ValueError(
#                 f"hidden_layers must be a non-empty list of positive ints, "
#                 f"got {hidden_layers}"
#             )

#         self.input_dim = int(input_dim)
#         self.output_dim = int(output_dim)
#         self.hidden_layers = list(hidden_layers)

#         initializer = keras.initializers.HeNormal(seed=seed)

#         model_layers = [keras.layers.Input(shape=(self.input_dim,))]
#         for n_units in self.hidden_layers:
#             model_layers.append(
#                 keras.layers.Dense(
#                     n_units, activation=activation, kernel_initializer=initializer
#                 )
#             )
#         # Output layer is LINEAR (no activation). Q-values can be any real
#         # number, including negative -- this env hands out penalties like
#         # -200 for obstacles. Squashing the output with relu/tanh would
#         # make negative Q-values impossible, which would be wrong.
#         model_layers.append(
#             keras.layers.Dense(
#                 self.output_dim, activation="linear", kernel_initializer=initializer
#             )
#         )

#         self.model = keras.Sequential(model_layers)
#         self.optimizer = keras.optimizers.Adam(
#             learning_rate=learning_rate,
#             clipnorm=grad_clip,   # caps how big one update step can be
#         )
#         self.loss_fn = keras.losses.MeanSquaredError()

#     # ------------------------------------------------------------------
#     def predict(self, x: np.ndarray) -> np.ndarray:
#         """x: shape (batch, input_dim) -> Q-values, shape (batch, output_dim)."""
#         x = np.atleast_2d(np.asarray(x, dtype=np.float32))
#         if x.shape[1] != self.input_dim:
#             raise ValueError(
#                 f"QNetwork expected input vectors of length {self.input_dim}, "
#                 f"but got length {x.shape[1]}. Check your observation encoding "
#                 f"or the 'input_dim' value in your config."
#             )
#         return self.model(x, training=False).numpy()

#     # ------------------------------------------------------------------
#     def train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
#         """
#         Nudge the network's predictions on `states` toward `targets`.

#         `targets` has the same shape as the network's output: (batch,
#         output_dim). Only the action actually taken should differ from
#         the network's own prediction -- everywhere else, target ==
#         prediction, so those entries contribute zero error and zero
#         gradient automatically. (Same trick as the numpy version, just
#         let TF differentiate it instead of us writing backward() by hand.)
#         """
#         states = np.asarray(states, dtype=np.float32)
#         targets = np.asarray(targets, dtype=np.float32)

#         with tf.GradientTape() as tape:
#             q_pred = self.model(states, training=True)
#             loss = self.loss_fn(targets, q_pred)

#         grads = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
#         return float(loss.numpy())

#     # ------------------------------------------------------------------
#     def copy_weights_from(self, other: "QNetwork") -> None:
#         """Used to sync the target network with the live Q-network."""
#         self.model.set_weights(other.model.get_weights())

#     def get_weights(self):
#         return self.model.get_weights()

#     def set_weights(self, weights) -> None:
#         self.model.set_weights(weights)


"""
predict() and train_step() are wrapped in tf.function.
Without this, TensorFlow re-validates and re-dispatches every operation
in plain eager Python mode on every call -- for a network this small,
that bookkeeping overhead dwarfs the actual math. tf.function traces the
computation into a static graph once and reuses it on every later call,
which is dramatically faster for small/frequent calls.
"""


class QNetwork:
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        output_dim: int,
        activation: str = "relu",
        learning_rate: float = 0.001,
        grad_clip: float = 10.0,
        seed: int = None,
    ):
        # input validation 
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not hidden_layers or any(h <= 0 for h in hidden_layers):
            raise ValueError(
                f"hidden_layers must be a non-empty list of positive ints, "
                f"got {hidden_layers}"
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = list(hidden_layers) # for later use store them 

        initializer = keras.initializers.HeNormal(seed=seed)

        model_layers = [keras.layers.Input(shape=(self.input_dim,))]
        for n_units in self.hidden_layers:
            model_layers.append(
                keras.layers.Dense(  # dense means every neuron connects to every other neuron
                    n_units, activation=activation, kernel_initializer=initializer
                )
            )
        # Linear output -- Q-values must be allowed to go negative
        # (this env hands out -200 obstacle penalties, etc).
        model_layers.append(
            keras.layers.Dense(
                self.output_dim, activation="linear", kernel_initializer=initializer
            )
        )

        self.model = keras.Sequential(model_layers) #tf assembles the nuerons in a nn through this
        self.optimizer = keras.optimizers.Adam( # optimiser tells how should weights be updated?
            learning_rate=learning_rate, clipnorm=grad_clip # caps how big one update step can be
        )
        self.loss_fn = keras.losses.MeanSquaredError() #MSE reduce error

        # ---- compile predict() and train_step() into static graphs ----
        # input_signature fixes the SHAPE (batch dim flexible via `None`,
        # so batch=1 from action-selection and batch=32 from training
        # both reuse the same compiled graph, traced only once each).
        self._predict_fn = tf.function( # reduces overhead by compiling one time and reusing it
            self._predict_impl,
            input_signature=[
                tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32) #none means batch size can change
            ],
        )
        self._train_step_fn = tf.function(
            self._train_step_impl,
            input_signature=[
                tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32),
                tf.TensorSpec(shape=[None, self.output_dim], dtype=tf.float32),
            ],
        )

    # ------------------------------------------------------------------
    # The "_impl" methods are the actual computation. The public
    # predict()/train_step() methods below handle numpy<->tensor
    # conversion and call the complied versions of these.
    # ------------------------------------------------------------------
    def _predict_impl(self, x):
        return self.model(x, training=False) # run a forward pass, no training just prediction 

    def _train_step_impl(self, states, targets):
        with tf.GradientTape() as tape:     # now training, computes derivatives automatically, does the math 
            q_pred = self.model(states, training=True) #does prediction
            loss = self.loss_fn(targets, q_pred)   # calculate loss
        grads = tape.gradient(loss, self.model.trainable_variables) # find gradient, which weights cause the error
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # apply gradient and update wghets
        return loss

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray: # just conversion into numpy array
        x = np.atleast_2d(np.asarray(x, dtype=np.float32)) # makes nested list
        if x.shape[1] != self.input_dim: 
            raise ValueError( # input validation
                f"QNetwork expected input vectors of length {self.input_dim}, "
                f"but got length {x.shape[1]}. Check your observation encoding "
                f"or the 'input_dim' value in your config."
            )
        return self._predict_fn(x).numpy() # calls the complied graph

    def train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        loss = self._train_step_fn(states, targets)
        return float(loss.numpy())  # same like predict but for train

    # ------------------------------------------------------------------
    def copy_weights_from(self, other: "QNetwork") -> None: # this is for the target netweork to copy 
        self.model.set_weights(other.model.get_weights())

    def get_weights(self):
        return self.model.get_weights() # returns all matrices and baises (this is what is saved in our pkl file)

    def set_weights(self, weights) -> None: #load those saved matrices back into the network
        self.model.set_weights(weights)