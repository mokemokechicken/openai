# https://qiita.com/Matchlab/items/18569ad3713d1c658da8

import gym
from gym import wrappers


# 正負でクラップしたPD制御のエージェント
class PDAgent:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def action(self, obs):
        m = self.kp * obs[2] + self.kd * obs[3]  # 操作量を計算
        if m >= 0:
            return 1
        if m < 0:
            return 0


video_path = './video'  # ビデオを保存するパス
n_episode = 1  # エピソード数
n_step = 200  # ステップ数
# PD制御パラメータ(ちなみP制御だけだと上手く行かない)
# ※正負でクラップしているので、比しか意味ない
kp = 0.1
kd = 0.01

my_agent = PDAgent(kp, kd)  # 正負でクラップしたPD制御のエージェント

env = gym.make('CartPole-v1')
env.reset()
# env = wrappers.Monitor(env, video_path, force=True)

for i_episode in range(n_episode):
    observation = env.reset()  # 環境初期化＆初期観察取得
    for t in range(n_step):
        env.render()  # 環境を表示(でもMonitorを使うとなくても表示される)
        # print(observation)
        action = my_agent.action(observation)  # エージェントクラスから行動を取得
        observation, reward, done, info = env.step(action)  # １ステップ進める
        if done:  # 終了フラグ
            print(f'Episode finished after {t+1} time steps')
            break
