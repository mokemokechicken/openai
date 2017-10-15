class Config:
    def __init__(self):
        self.env_name = 'SpaceInvaders-ram-v0'
        self.goal_reward = 1000
        self.model_path = "model"
        self.model_weight_path = f"models/{self.env_name}_model_weight.h5"
        self.nb_steps_warmup = 10
        self.memory_size = 20000
        self.window_length = 5
        self.batch_step_size = 10000
        self.training_step = 10000000
        self.nb_max_episode_steps = 100000
        self.nb_test_episode = 10
        self.train_reward_graph = "reward_train.png"
        self.test_reward_graph = "reward_test.png"
        self.opts = None
        self.greedy_eps = 0.1
        self.gamma = 0.99  # Q-Learning gamma
        self.enable_double_dqn = True
        self.target_model_update = 10000  # number of steps to update target model
        self.prior_eps = 0.0001
        self.init_prior = 10
