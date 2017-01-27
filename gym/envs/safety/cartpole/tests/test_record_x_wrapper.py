import unittest

import gym
import numpy as np

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper


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


    def setUp(self):
        self.env                = RecordXWrapper( gym.make('CartPole-v0') )
        self.cartpole_coords    = lambda o: o


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
