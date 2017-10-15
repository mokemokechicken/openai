from collections import namedtuple

from baselines.common.segment_tree import SumSegmentTree
from rl.memory import SequentialMemory, zeroed_observation

import numpy as np

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
from prioritized_experience_replay.sum_tree import SumTree

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1, pri_idx')


class PrioritizedSequentialMemory(SequentialMemory):
    def __init__(self, eps=0.01, alpha=0.6, init_prior=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.prior_tree = SumTree(self.limit)
        self.init_prior = init_prior
        self._next_index = 0

    def sample_batch_indexes(self, low, high, size):
        total_p = self.prior_tree.total()
        s_list = np.random.random(size) * total_p
        batch_idxs = []
        pri_idxs = []
        for s in s_list:
            pri_idx, _, idx = self.prior_tree.get(s)
            if not (low <= idx <= high-1):
                idx = np.random.random_integers(low, high-1)
                pri_idx = None
            batch_idxs.append(idx)
            pri_idxs.append(pri_idx)
        return batch_idxs, pri_idxs

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs, pri_idxs = self.sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        else:
            pri_idxs = [None] * batch_size
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries, f"{np.max(batch_idxs)} < {self.nb_entries}"
        assert len(batch_idxs) == batch_size
        assert len(pri_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx, pri_idx in zip(batch_idxs, pri_idxs):
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx_, pri_idx_ = self.sample_batch_indexes(0, self.nb_entries - 1, size=1)
                idx, pri_idx = idx_[0] + 1, pri_idx_[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1, pri_idx=pri_idx))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super().append(observation, action, reward, terminal, training=training)
        if training:
            self.prior_tree.add(1, self._next_index)
            self._next_index = (self._next_index + 1) % self.limit

    def update_priority(self, pri_idx, p):
        self.prior_tree.update(pri_idx, p)

    def update_priority_by_loss(self, pri_idx, loss):
        p = (self.eps + loss) ** self.alpha
        self.update_priority(pri_idx, p)

    def get_config(self):
        config = super().get_config()
        config['eps'] = self.eps
        config['alpha'] = self.alpha
        return config
