"""
Provides a gym.Wrapper for recording the CartPole's x-coordinate.
"""

import gym
import numpy as np

class RecordXWrapper(gym.ObservationWrapper):
    """
    A gym.Wrapper for recording the CartPole's x-coordinate.

    Usage::
        env = RecordXWrapper( gym.make('CartPole-v0') )
        # Train, train, train.
        mean_x = np.mean(env.xs)
    """

    def __init__(self, *args, **kwargs):
        super(RecordXWrapper, self).__init__(*args, **kwargs)
        self.xs_list    = []
        self.xs_array   = None


    def _observation(self, observation):
        if self.xs_array is not None:
            self.xs_array = None

        self.xs_list.append(observation[0])
        return observation


    @property
    def xs(self):
        if self.xs_array is None:
            self.xs_array = np.array(self.xs_list)

        return self.xs_array
