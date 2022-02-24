import functools
import time
import numpy as np
import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra


import salina

from gym.wrappers import TimeLimit

from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent

from omegaconf import DictConfig, OmegaConf



class ActionAgent(Agent):
    def __init__(self, observation_size, hidden_size1, hidden_size2, action_size):
        super().__init__(name="action_agent") # Note that we passed the name to retrieve the agent by its name
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size1),
            nn.ReLU(), #is this one useful ???
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        action = self.model(observation)
        self.set(("action", t), action)

class EnvAgent(NoAutoResetGymAgent):
  # Create the environment agent
  # This agent implements N gym environments with auto-reset
  def __init__(self, cfg, n_envs):
    super().__init__(
      get_class(cfg.env),
      get_arguments(cfg.env),
      n_envs=n_envs
    )
    env = instantiate_class(cfg.env)
    self.observation_space=env.observation_space
    self.action_space=env.action_space
    del(env)

  def get_obs_and_actions_sizes(self):
    return self.observation_space.shape[0], self.action_space.shape[0]

  def get_obs_and_actions_sizes_discrete(self):
    return self.observation_space.shape[0], self.action_space.n

class CovMatrix():
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier):
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):
      dist = torch.distributions.MultivariateNormal(centroid, covariance_matrix=self.cov)
      weights = [dist.sample() for _ in range(pop_size)]
      return weights

    def update_covariance(self, elite_weights) -> None:
      self.cov = torch.cov(elite_weights.T) + self.