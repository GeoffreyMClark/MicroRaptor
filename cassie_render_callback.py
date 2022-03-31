import os
import cv2
import gym

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class RenderCallback(BaseCallback):

    def __init__(self, render_freq: int, env, verbose: int = 1):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq
        self.env=env

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            for _ in range(1):
                obs = self.env.reset()
                done= False
                while done == False:
                    action, _states = self.model.predict(obs)
                    obs, reward, done, info = self.env.step(action)
                    raw=self.env.render(mode='rgb_array')
                    cv2.imshow("cassie_standing_model",raw)
                    cv2.waitKey(1)
