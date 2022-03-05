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
import salina.rl.functional as RLF
from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent, NRemoteAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent, AutoResetGymAgent
from salina.agents.asynchronous import AsynchronousAgent

from omegaconf import DictConfig, OmegaConf

from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
from salina_examples import weight_init

import torch.multiprocessing as mp

from agents import QMLPAgent, make_gym_env, ActionMLPAgent


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
      self.cov = torch.cov(elite_weights.T) + self.noise


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def set_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)

def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

def to_numpy(var):
    return var.data.numpy()

def run_cem_td3(q_agent_1, q_agent_2, action_agent, logger, cfg):






  # 1)  Build the  logger
  torch.manual_seed(cfg.algorithm.env_seed)
  logger = instantiate_class(cfg.logger)
  
  pop_size = cfg.algorithm.pop_size

  temporal_agents=[]
  for _ in range(cfg.algorithm.n_processes): 
    env_agent = EnvAgent(cfg,n_envs=cfg.algorithm.n_envs)
    observation_size, action_size = env_agent.get_obs_and_actions_sizes()
    action_agent = ActionAgent(observation_size, cfg.algorithm.architecture.hidden_size1, cfg.algorithm.architecture.hidden_size2, action_size)  
    temporal_agent=TemporalAgent(Agents(env_agent, action_agent))
    temporal_agent.seed(cfg.algorithm.env_seed)
    agent=AsynchronousAgent(temporal_agent)
    temporal_agents.append(agent)
  
  # This agent is used to convert parameters to vectors since the vector_to_parameters does not seem to work when moduule are in different processes
  env_agent = EnvAgent(cfg,n_envs=cfg.algorithm.n_envs)
  observation_size, action_size = env_agent.get_obs_and_actions_sizes()    
  action_agent = ActionAgent(observation_size, cfg.algorithm.architecture.hidden_size1, cfg.algorithm.architecture.hidden_size2, action_size)  
  temporal_agent=TemporalAgent(Agents(env_agent, action_agent))

  centroid = torch.nn.utils.parameters_to_vector(temporal_agent.parameters())
  matrix = CovMatrix(centroid, cfg.algorithm.sigma, 
                     cfg.algorithm.noise_multiplier, 
                     )

  best_score = -np.inf


  #TD3
  q_target_agent_1 = copy.deepcopy(q_agent_1)
  q_target_agent_2 = copy.deepcopy(q_agent_2)
  action_target_agent = copy.deepcopy(action_agent)

  acq_action_agent = copy.deepcopy(action_agent)
  acq_agent = TemporalAgent(Agents(env_agent, acq_action_agent))
  acq_remote_agent, acq_workspace = NRemoteAgent.create(
      acq_agent,
      num_processes=cfg.algorithm.n_processes,
      t=0,
      n_steps=cfg.algorithm.n_timesteps,
      epsilon=1.0,
  ) 
  acq_remote_agent.seed(cfg.algorithm.env_seed)

  # == Setting up the training agents
  train_temporal_q_agent_1 = TemporalAgent(q_agent_1)
  train_temporal_q_agent_2 = TemporalAgent(q_agent_2)
  train_temporal_action_agent = TemporalAgent(action_agent)
  train_temporal_q_target_agent_1 = TemporalAgent(q_target_agent_1)
  train_temporal_q_target_agent_2 = TemporalAgent(q_target_agent_2)
  train_temporal_action_target_agent = TemporalAgent(action_target_agent)

  train_temporal_q_agent_1.to(cfg.algorithm.loss_device)
  train_temporal_q_agent_2.to(cfg.algorithm.loss_device)
  train_temporal_action_agent.to(cfg.algorithm.loss_device)
  train_temporal_q_target_agent_1.to(cfg.algorithm.loss_device)
  train_temporal_q_target_agent_2.to(cfg.algorithm.loss_device)
  train_temporal_action_target_agent.to(cfg.algorithm.loss_device)

  acq_remote_agent(
    acq_workspace,
    t=0,
    n_steps=cfg.algorithm.n_timesteps,
      epsilon=cfg.algorithm.action_noise,
  )

  # == Setting up & initializing the replay buffer for DQN
  replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
  # replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)
  # """Replay Buffer bien initialise ou pas ?"""

  # logger.message("[DDQN] Initializing replay buffer")
  # while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
  #     acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
  #     acq_remote_agent(
  #         acq_workspace,
  #         t=cfg.algorithm.overlapping_timesteps,
  #         n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
  #         epsilon=cfg.algorithm.action_noise,
  #     )
  #     replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)
  # """Initialisation replay buffer jusqu'ici"""

  logger.message("[DDQN] Learning")
  n_interactions = 0
  optimizer_args = get_arguments(cfg.algorithm.optimizer)
  optimizer_q_1 = get_class(cfg.algorithm.optimizer)(
      q_agent_1.parameters(), **optimizer_args
  )
  optimizer_q_2 = get_class(cfg.algorithm.optimizer)(
      q_agent_2.parameters(), **optimizer_args
  )
  optimizer_action = get_class(cfg.algorithm.optimizer)(
      action_agent.parameters(), **optimizer_args
  )
  iteration = 0

  #TD3



  """TD3 implemente jusqu'ici"""



  # 7) Training loop
  for epoch in range(cfg.algorithm.max_epochs):
    matrix.update_noise()
    scores = []
    weights = matrix.generate_weights(centroid, pop_size) 

    """introduce td3 here :"""
    
    if (epoch > cfg.algorithm.init_cem) : 
      """Boucle For TD3"""
      for inner_epoch in range(cfg.algorithm.inner_epochs):
        batch_size = cfg.algorithm.batch_size
        replay_workspace = replay_buffer.get(batch_size).to(
          cfg.algorithm.loss_device
        )
        done, reward = replay_workspace["env/done", "env/reward"]
        train_temporal_q_agent_1(
          replay_workspace,
          t=0,
          n_steps=cfg.algorithm.buffer_time_size,
          detach_action=True,
        )
        q_1 = replay_workspace["q"].squeeze(-1)
        train_temporal_q_agent_2(
          replay_workspace,
          t=0,
          n_steps=cfg.algorithm.buffer_time_size,
          detach_action=True,
        )
        q_2 = replay_workspace["q"].squeeze(-1)

        with torch.no_grad():
          train_temporal_action_target_agent(
            replay_workspace,
            t=0,
            n_steps=cfg.algorithm.buffer_time_size,
            epsilon=cfg.algorithm.target_noise,
            epsilon_clip=cfg.algorithm.noise_clip,
          )
          train_temporal_q_target_agent_1(
            replay_workspace,
            t=0,
            n_steps=cfg.algorithm.buffer_time_size,
          )
          q_target_1 = replay_workspace["q"]
          train_temporal_q_target_agent_2(
            replay_workspace,
            t=0,
            n_steps=cfg.algorithm.buffer_time_size,
          )
          q_target_2 = replay_workspace["q"]

        q_target = torch.min(q_target_1, q_target_2).squeeze(-1)
        target = (
          reward[1:]
          + cfg.algorithm.discount_factor
          * (1.0 - done[1:].float())
          * q_target[1:]
        )

        td_1 = q_1[:-1] - target
        td_2 = q_2[:-1] - target
        error_1 = td_1 ** 2
        error_2 = td_2 ** 2

        burning = torch.zeros_like(error_1)
        burning[cfg.algorithm.burning_timesteps :] = 1.0
        error_1 = error_1 * burning
        error_2 = error_2 * burning
        error = error_1 + error_2
        loss = error.mean()
        logger.add_scalar("loss/td_loss_1", error_1.mean().item(), iteration)
        logger.add_scalar("loss/td_loss_2", error_2.mean().item(), iteration)
        optimizer_q_1.zero_grad()
        optimizer_q_2.zero_grad()
        loss.backward()

        if cfg.algorithm.clip_grad > 0:
          n = torch.nn.utils.clip_grad_norm_(
            q_agent_1.parameters(), cfg.algorithm.clip_grad
          )
          logger.add_scalar("monitor/grad_norm_q_1", n.item(), iteration)
          n = torch.nn.utils.clip_grad_norm_(
            q_agent_2.parameters(), cfg.algorithm.clip_grad
          )
          logger.add_scalar("monitor/grad_norm_q_2", n.item(), iteration)

        optimizer_q_1.step()
        optimizer_q_2.step()

        if inner_epoch % cfg.algorithm.policy_delay:
          train_temporal_action_agent(
            replay_workspace,
              epsilon=0.0,
              t=0,
              n_steps=cfg.algorithm.buffer_time_size,
          )
          train_temporal_q_agent_1(
            replay_workspace,
            t=0,
            n_steps=cfg.algorithm.buffer_time_size,
          )
          q = replay_workspace["q"].squeeze(-1)
          burning = torch.zeros_like(q)
          burning[cfg.algorithm.burning_timesteps :] = 1.0
          q = q * burning
          q = q * (1.0 - done.float())
          optimizer_action.zero_grad()
          loss = -q.mean()
          """loss to use"""
          td3_loss = loss
          loss.backward()

          if cfg.algorithm.clip_grad > 0:
            n = torch.nn.utils.clip_grad_norm_(
              action_agent.parameters(), cfg.algorithm.clip_grad
            )
            logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)

          logger.add_scalar("loss/q_loss", loss.item(), iteration)
          optimizer_action.step()

          tau = cfg.algorithm.update_target_tau
          soft_update_params(q_agent_1, q_target_agent_1, tau)
          soft_update_params(q_agent_2, q_target_agent_2, tau)
          soft_update_params(action_agent, action_target_agent, tau)

        iteration += 1
      
      # if epoch % cfg.algorithm.td3_update_modulo: 
      
      #   #for idx_agent in range(cfg.algorithm.n_processes):
      #   #set_params(action_agent, temporal_agents[idx_agent].agent.agent.agents[1], tau)
        
        
      #   print("loss td3", td3_loss)
      #   for pop in range(pop_size // 2):
      #     # if td3_loss < 0 : 
      #     #   weights[pop] *= -td3_loss
      #     # else :
      #     #   weights[pop] *= td3_loss
      #     weights[pop] = torch.tensor(action_agent.parameters())
      #   logger.message("TD3 Pop Introduction")
      
    
    #CEM
    position=0
    while(position<pop_size):
      n_to_launch=min(pop_size-position,cfg.algorithm.n_processes)
      # Execute the agents
      for idx_agent in range(n_to_launch):        
        idx_weight=idx_agent+position
        torch.nn.utils.vector_to_parameters(weights[idx_weight], temporal_agent.parameters())
        temporal_agents[idx_agent].agent.load_state_dict(temporal_agent.state_dict())
        if (epoch % cfg.algorithm.td3_update_modulo and epoch > cfg.algorithm.init_cem and position < pop_size // 2 ):
          soft_update_params(action_agent, temporal_agents[idx_agent].agent.agent.agents[1], tau)
        temporal_agents[idx_agent](t=0,stop_variable="env/done")
      
      #Wait for agents execution
      running=True
      while running:
        are_running=[a.is_running() for a in temporal_agents[:n_to_launch]]
        running=any(are_running)

      #Compute collected reward
      workspaces=[a.get_workspace() for a in temporal_agents[:n_to_launch]]
      for k,workspace in enumerate(workspaces):      
        episode_lengths=workspace["env/done"].float().argmax(0)+1
        arange=torch.arange(cfg.algorithm.n_envs) 
        mean_reward=workspace["env/cumulated_reward"][episode_lengths-1,arange].mean().item()
        scores.append(mean_reward)
        if mean_reward >= best_score:
          best_score = mean_reward
          best_params=weights[k+position]
    
        if cfg.verbose > 0: 
          print(f"Indiv: {k+position + 1} score {scores[k+position]:.2f}")
        

        replay_buffer.put(workspace, cfg.algorithm.buffer_time_size)
        """mettre replay buffer addition ici"""
        
      position+=n_to_launch

    print("Best score: ", best_score)
    # Keep only best individuals to compute the new centroid
    elites_idxs=np.argsort(scores)[-cfg.algorithm.elites_nb :]
    elites_weights = [weights[k] for k in elites_idxs]
    elites_weights = torch.cat([torch.tensor(w).unsqueeze(0) for w in elites_weights],dim=0)
    centroid = elites_weights.mean(0)

    # Update covariance
    matrix.update_noise()
    matrix.update_covariance(elites_weights)
  
  for a in temporal_agents:
    a.close()

@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    #cfg=OmegaConf.load("gym.yaml")
    mp.set_start_method("spawn")
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    #q_agent_1 = instantiate_class(cfg.q_agent)
    q_agent_1 = QMLPAgent(**cfg.q_agent)
    #q_agent_2 = instantiate_class(cfg.q_agent)
    q_agent_2 = QMLPAgent(**cfg.q_agent)
    q_agent_2.apply(weight_init)
    action_agent = ActionMLPAgent(**cfg.action_agent)
    run_cem_td3(q_agent_1, q_agent_2, action_agent, logger, cfg)

if __name__ == "__main__":
    main()