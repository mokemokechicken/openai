class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.model_path = "model"
        self.model_weight_path = "model_weight.h5"
        self.memory_size = 20000
        self.window_length = 2
        self.training_step = 20000
        self.nb_max_episode_steps = 300
        self.visualize_training = False
        self.visualize_test = True
        self.nb_test_episode = 10
