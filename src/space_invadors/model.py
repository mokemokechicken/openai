from keras.engine.topology import Input
from keras.layers.core import Dense, Flatten
from keras.layers.merge import Add
from rl.keras_future import Model


def build_model(env, config):
    """

    :param Config config:
    :param gym.core.Env env:
    :return:
    """
    n_dims = [128, 128, 64]

    in_x = x = Input(shape=(config.window_length,) + env.observation_space.shape)
    x = Flatten()(x)
    for n_dim in n_dims:
        x = Dense(n_dim, activation="relu")(x)
    x = Dense(env.action_space.n, activation="linear")(x)
    model = Model(input=in_x, output=x)
    return model


def add_resnet(x, n_dim):
    in_x = x
    x = Dense(n_dim, activation="relu")(x)
    x = Dense(n_dim, activation="relu")(x)
    return Add()([in_x, x])
