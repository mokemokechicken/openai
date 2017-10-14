import os
from argparse import ArgumentParser

import gym
from gym.core import Wrapper
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy

from mountain_car.model import build_model
from mountain_car.config import Config

import numpy as np
import matplotlib.pyplot as plt


class WrapEnv(Wrapper):
    def _step(self, action):
        observation, reward, done, info = super()._step(action)
        # observation: (position, velocity), reward = always -1
        # reward = -1 + abs(observation[1])
        return observation, reward, done, info


def create_agent(config: Config, env, model):
    memory = SequentialMemory(limit=config.memory_size, window_length=config.window_length)

    # 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
    policy = EpsGreedyQPolicy(eps=config.greedy_eps)
    # policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=config.nb_steps_warmup,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def train(config: Config):
    env = gym.make(config.env_name)
    # env = WrapEnv(env)
    env.reset()
    # env = wrappers.Monitor(env, video_path, force=True)

    model = build_model(env, config)
    dqn = create_agent(config, env, model)
    if not config.opts.new and os.path.exists(config.model_weight_path):
        dqn.load_weights(config.model_weight_path)

    reward_history = []
    num_steps = 0
    while num_steps < config.training_step:
        num_steps += config.batch_step_size
        history = dqn.fit(env, nb_steps=config.batch_step_size, visualize=config.opts.render,
                          verbose=get_verbose_level(config),
                          nb_max_episode_steps=config.nb_max_episode_steps)
        reward_history += history.history.get("episode_reward")
        last_100_average = np.average(reward_history[-100:])
        print(f"total episode={len(reward_history)}, last 100 episode reward average={last_100_average}")
        if len(reward_history) >= 100 and last_100_average > config.goal_reward:
            break
        dqn.policy.eps = config.greedy_eps * (1 - num_steps / config.training_step)
        model.save_weights(config.model_weight_path, overwrite=True)

    if reward_history:
        plt.plot(reward_history)
        plt.savefig(config.train_reward_graph)

    print(f"save model to {config.model_weight_path}")
    model.save_weights(config.model_weight_path, overwrite=True)


def get_verbose_level(config):
    ret = 1
    if config.opts.verbose:
        ret = 2
    return ret


def evaluate(config: Config):
    env = gym.make(config.env_name)
    env.reset()
    env._max_episode_steps = config.nb_max_episode_steps
    # env = wrappers.Monitor(env, video_path, force=True)

    model = build_model(env, config)
    dqn = create_agent(config, env, model)
    dqn.load_weights(config.model_weight_path)
    history = dqn.test(env, nb_episodes=config.nb_test_episode, visualize=config.opts.render)
    rewards = history.history.get("episode_reward")
    steps = history.history.get('nb_episode_steps')
    if rewards:
        print(f"reward: mean={np.mean(rewards)}, std={np.std(rewards)}")
    if steps:
        print(f"steps: mean={np.mean(steps)}, std={np.std(steps)}")


def main():
    parser = get_opt_parser()
    opts = parser.parse_args()

    target = opts.target
    config = Config()
    config.opts = opts

    if opts.episode:
        config.nb_test_episode = opts.episode

    if target == "train":
        train(config)
    elif target == "test":
        evaluate(config)


def get_opt_parser():
    parser = ArgumentParser()
    parser.add_argument("target", choices=["train", "test"])
    parser.add_argument("--new", dest="new", action="store_true", help="create new model")
    parser.add_argument("--render", dest="render", action="store_true", help="render or not")
    parser.add_argument("-v", dest="verbose", action="store_true", help="verbose mode")
    parser.add_argument("--ep", dest="episode", type=int, help="number of episode to test")
    return parser


if __name__ == '__main__':
    main()
