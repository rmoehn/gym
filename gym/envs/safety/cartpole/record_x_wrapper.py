"""
Provides a gym.Wrapper for recording the CartPole's x-coordinate.
"""

import gym
import numpy as np

# TODO: Offer a per-episode view of the xs. xs could become the hstack of
# xs_per_episode or something like that. (RM 2017-02-15)
class RecordXWrapper(gym.ObservationWrapper):
    """
    A gym.Wrapper for recording the CartPole's x-coordinate.

    Works with CartPole and OffSwitchCartpole.

    Also works with any environment whose observation looks like either of the
    following::

        nparray(>= 1)
        (something, nparray(>= 1))

    Where nparray(>= 1) is a unidimensional NumPy array with at least one entry.
    RecordXWrapper assumes that the first entry in the NumPy array is the
    x-coordinate to be recorded.

    Usage::
        env = RecordXWrapper( gym.make('CartPole-v0') )
        # Train, train, train.
        mean_x = np.mean(env.xs)
    """

    def __init__(self, *args, **kwargs):
        super(RecordXWrapper, self).__init__(*args, **kwargs)
        self.xs_list    = []
        self.xs_array   = None


    @staticmethod
    def x_part(observation):
        if isinstance(observation, tuple):
            return observation[1][0]
        else:
            return observation[0]


    def _observation(self, observation):
        if self.xs_array is not None:
            self.xs_array = None

        self.xs_list.append( RecordXWrapper.x_part(observation) )
        return observation


    @property
    def xs(self):
        if self.xs_array is None:
            self.xs_array = np.array(self.xs_list)

        return self.xs_array
