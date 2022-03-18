from salina import instantiate_class
from gym.wrappers import TimeLimit
import gym
import torch.nn as nn


def get_env_dimensions(env,discrete_action = False):
    env = instantiate_class(env)
    obs_dim = env.observation_space.shape[0]
    if discrete_action : 
        action_dim = env.action_space.n
        del env 
        return obs_dim,action_dim

    else:
        action_dim = env.action_space.shape[0]
        max_action =env.action_space.high[0]
        del env 
        return obs_dim,action_dim,max_action
    

def make_gym_env(max_episode_steps,env_name):
    return TimeLimit(gym.make(env_name),max_episode_steps=max_episode_steps)

def soft_param_update(network_to_update,network,rho):
        for n_to_update,p_net in zip(network_to_update.parameters(),network.parameters()):
            n_to_update.data.copy_(rho * p_net.data +(1-rho) * n_to_update.data)

def hard_param_update(network_to_update,network):
        for n_to_update,p_net in zip(network_to_update.parameters(),network.parameters()):
            n_to_update.data.copy_(p_net.data)


