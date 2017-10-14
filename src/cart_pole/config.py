class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.model_path = "model"
        self.model_weight_path = "model_weight.h5"
        self.memory_size = 50000
        self.window_length = 2
        self.training_step = 50000
        self.nb_max_episode_steps = 500
        self.visualize_training = True
        self.visualize_test = True
        self.nb_test_episode = 10
        self.train_reward_graph = "reward_train.png"
        self.test_reward_graph = "reward_test.png"
