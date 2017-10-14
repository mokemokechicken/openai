from keras.engine.topology import Input
from keras.layers.core import Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from rl.keras_future import Model
from tensorflow.contrib.rnn.python.ops.rnn_cell import LayerNormBasicLSTMCell


def build_model(env, config):
    """

    :param Config config:
    :param gym.core.Env env:
    :return:
    """
    n_dim = 64
    n_layer = 2

    in_x = x = Input(shape=(config.window_length,) + env.observation_space.shape)
    x = Flatten()(x)
    x = Dense(n_dim, activation="relu")(x)
    for _ in range(n_layer):
        x = add_resnet(x, n_dim)
    x = Dense(env.action_space.n, activation="linear")(x)
    model = Model(input=in_x, output=x)
    return model


def add_resnet(x, n_dim):
    in_x = x
    x = Dense(n_dim, activation="relu")(x)
    return Add()([in_x, x])
