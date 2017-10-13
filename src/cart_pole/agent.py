from keras.engine.topology import Input
from keras.layers.core import Dense, Flatten
from rl.keras_future import Model

from cart_pole.config import Config


def build_model(env, config: Config):
    """

    :param gym.wrappers.time_limit.TimeLimit env:
    :return:
    """
    in_x = x = Input(shape=(config.window_length,) + env.observation_space.shape)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(env.action_space.n)(x)
    model = Model(input=in_x, output=x)
    return model

