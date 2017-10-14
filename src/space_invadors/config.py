class Config:
    def __init__(self):
        self.env_name = 'SpaceInvaders-ram-v0'
        self.goal_reward = 1000
        self.model_path = "model"
        self.model_weight_path = "model_weight.h5"
        self.nb_steps_warmup = 100
        self.memory_size = 1000000
        self.window_length = 4
        self.batch_step_size = 10000
        self.training_step = 1000000
        self.nb_max_episode_steps = 10000
        self.nb_test_episode = 10
        self.train_reward_graph = "reward_train.png"
        self.test_reward_graph = "reward_test.png"
        self.opts = None
        self.greedy_eps = 0.05
