import os
from argparse import ArgumentParser

import gym

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy

from prioritized_experience_replay.prioritized_dqn_agent import PrioritizedDQNAgent
from prioritized_experience_replay.prioritized_memory import PrioritizedSequentialMemory
from space_invadors.model import build_model
from space_invadors.config import Config

import numpy as np
import matplotlib.pyplot as plt


class InvadorProcessor(Processor):
    def __init__(self, training=True):
        self.training = training
        self.dead = False
        self.mode = 0

    def process_observation(self, observation):
        mode = observation[-19]
        if self.mode != 4 and mode == 4:  # dead state
            self.dead = True
        self.mode = mode
        return observation / 255

    def process_reward(self, reward):
        if self.training:
            reward = reward / 100
            if self.dead and self.mode == 4:
                reward = -1
                self.dead = False
            return reward
        else:
            return reward


def create_agent(config: Config, env, model, training=True):
    # memory = SequentialMemory(limit=config.memory_size, window_length=config.window_length)
    memory = PrioritizedSequentialMemory(limit=config.memory_size, window_length=config.window_length,
                                         eps=config.prior_eps, init_prior=config.init_prior)

    policy = EpsGreedyQPolicy(eps=config.greedy_eps)
    # if not training:
    #     policy = BoltzmannQPolicy(tau=0.01)
    processor = InvadorProcessor(training=training)
    nb_steps_warmup = config.nb_steps_warmup if training else 0

    # dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=nb_steps_warmup,
    #                target_model_update=config.target_model_update, policy=policy, test_policy=policy, processor=processor,
    #                gamma=config.gamma, enable_double_dqn=config.enable_double_dqn,
    #                )
    dqn = PrioritizedDQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
                              nb_steps_warmup=nb_steps_warmup, target_model_update=config.target_model_update,
                              policy=policy, test_policy=policy, processor=processor,
                              gamma=config.gamma, enable_double_dqn=config.enable_double_dqn
                              )

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def train(config: Config):
    env = gym.make(config.env_name)
    env.reset()

    model = build_model(env, config)
    dqn = create_agent(config, env, model, training=True)
    if not config.opts.new and os.path.exists(config.model_weight_path):
        print(f"loading model: {config.model_weight_path}")
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
        dqn.save_weights(config.model_weight_path, overwrite=True)

    if reward_history:
        plt.plot(reward_history)
        plt.savefig(config.train_reward_graph)

    print(f"save model to {config.model_weight_path}")
    dqn.save_weights(config.model_weight_path, overwrite=True)


def get_verbose_level(config):
    ret = 1
    if config.opts.verbose:
        ret = 2
    return ret


def evaluate(config: Config):
    env = gym.make(config.env_name)
    env.reset()
    #print(env.observation_space)
    #print(env.observation_space.high)
    #print(env.observation_space.low)

    model = build_model(env, config)
    dqn = create_agent(config, env, model, training=False)
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
