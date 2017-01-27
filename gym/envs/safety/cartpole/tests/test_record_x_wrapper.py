"""Test for RecordXWrapper on CartPole and OffSwitchCartpole."""

import operator
import unittest

import gym
import numpy as np

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper

# I will subclass this later, overriding setUp in order to test RecordXWrapper
# with OffSwitchCartpole, too. This is not a pretty way of doing it, but I think
# there is no pretty way unless you use plugins with nose2.
class TestRecordXWrapper(unittest.TestCase):
    def run_sample(self, n_episodes, max_steps):
        xs = []

        for _ in xrange(n_episodes):
            observation = self.env.reset()
            xs.append( self.cartpole_coords(observation)[0] )
            done        = False

            for _ in xrange(max_steps):
                action = self.env.action_space.sample()
                observation, _, done, _ = self.env.step(action)
                xs.append( self.cartpole_coords(observation)[0] )

                if done:
                    break

        return xs


    # Note: unittest's setUp design is a bit awful. Think of this as providing
    # two extra arguments -- env and cartpole_coords -- to run_sample in every
    # test.
    def setUp(self):
        self.env                = RecordXWrapper( gym.make('CartPole-v0') )
        self.cartpole_coords    = lambda o: o
            # Map whatever the observations the environment returns to standard
            # CartPole observations.


    def test_until_done(self):
        xs = self.run_sample(n_episodes=5, max_steps=30)
        self.assertTrue( np.all( xs == self.env.xs ) )


    def test_before_done(self):
        xs = self.run_sample(n_episodes=5, max_steps=7)
        self.assertTrue( np.all( xs == self.env.xs ) )


    def test_two_reads(self):
        self.run_sample(n_episodes=5, max_steps=30)
        self.assertTrue( np.all( self.env.xs == self.env.xs ) )


    def test_reuptake(self):
        xs0 = self.run_sample(n_episodes=5, max_steps=30)
        xs1 = self.run_sample(n_episodes=5, max_steps=7)

        self.assertTrue( np.all( xs0 + xs1 == self.env.xs ) )


class TestRecordXWrapperOffSwitchCartpole(TestRecordXWrapper):
    def setUp(self):
        self.env = RecordXWrapper( gym.make('OffSwitchCartpole-v0') )
        self.cartpole_coords = operator.itemgetter(1)
