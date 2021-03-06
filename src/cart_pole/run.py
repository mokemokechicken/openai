from pprint import pprint

import gym
from gym import wrappers
from gym.core import Wrapper
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy

from cart_pole.model import build_model
from cart_pole.config import Config

import matplotlib.pyplot as plt
import numpy as np


class WrapEnv(Wrapper):
    def _step(self, action):
        observation, reward, done, info = super()._step(action)
        # observation: (x,x_dot,theta,theta_dot)
        reward -= observation[0] * observation[0] + observation[2] * observation[2]
        return observation, reward, done, info


def create_agent(config: Config, env, model):
    memory = SequentialMemory(limit=config.memory_size, window_length=config.window_length)
    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    # policy = EpsGreedyQPolicy(eps=0.1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def train(config: Config):
    env = gym.make(config.env_name)
    env = WrapEnv(env)
    env.reset()
    # env = wrappers.Monitor(env, video_path, force=True)

    model = build_model(env, config)
    dqn = create_agent(config, env, model)

    history = dqn.fit(env, nb_steps=config.training_step, visualize=config.visualize_training, verbose=2,
                      nb_max_episode_steps=config.nb_max_episode_steps)

    # rewards = history.history.get("episode_reward")
    steps = history.history.get('nb_episode_steps')
    if steps:
        plt.plot(steps)
        plt.savefig(config.train_reward_graph)

    print(f"save model to {config.model_weight_path}")
    model.save_weights(config.model_weight_path, overwrite=True)


def evaluate(config: Config):
    env = gym.make(config.env_name)
    env.reset()
    # env = wrappers.Monitor(env, video_path, force=True)

    model = build_model(env, config)
    dqn = create_agent(config, env, model)
    dqn.load_weights(config.model_weight_path)
    history = dqn.test(env, nb_episodes=config.nb_test_episode, visualize=config.visualize_test)
    rewards = history.history.get("episode_reward")
    steps = history.history.get('nb_episode_steps')
    if rewards:
        print(f"reward: mean={np.mean(rewards)}, std={np.std(rewards)}")
    if steps:
        print(f"steps: mean={np.mean(steps)}, std={np.std(steps)}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: run.py train|eval")
        exit()

    target = sys.argv[1]
    config = Config()
    if target == "train":
        train(config)
    elif target == "eval":
        evaluate(config)
    else:
        print(f"unknown command {target}")


if __name__ == '__main__':
    main()
