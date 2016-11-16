import unittest

import gym
import numpy as np

from gym.envs.safety.cartpole.record_x_wrapper import RecordXWrapper

def run_sample(env, n_episodes, max_steps):
    xs = []

    for _ in xrange(n_episodes):
        observation = env.reset()
        xs.append(observation[0])
        done        = False

        for _ in xrange(max_steps):
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)
            xs.append(observation[0])

            if done:
                break

    return xs


class TestRecordXWrapper(unittest.TestCase):
    def setUp(self):
        self.env = RecordXWrapper( gym.make('CartPole-v0') )


    def test_until_done(self):
        xs = run_sample(self.env, n_episodes=5, max_steps=30)
        self.assertTrue( np.all( xs == self.env.xs ) )


    def test_before_done(self):
        xs = run_sample(self.env, n_episodes=5, max_steps=7)
        self.assertTrue( np.all( xs == self.env.xs ) )


    def test_two_reads(self):
        run_sample(self.env, n_episodes=5, max_steps=30)
        self.assertTrue( np.all( self.env.xs == self.env.xs ) )


    def test_reuptake(self):
        xs0 = run_sample(self.env, n_episodes=5, max_steps=30)
        xs1 = run_sample(self.env, n_episodes=5, max_steps=7)

        self.assertTrue( np.all( xs0 + xs1 == self.env.xs ) )
