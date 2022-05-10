import sys
import os
import gym
import hydra
import my_gym
import torch
import tqdm
from gym.wrappers import TimeLimit
from omegaconf import DictConfig
from salina import instantiate_class,Workspace
from salina.agents import Agents,TemporalAgent,NRemoteAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.agents.asynchronous import AsynchronousAgent
from time import time
import csv

sys.path.append(os.getcwd())
from algorithms.cem_erl import CemERl


HYDRA_FULL_ERROR=1


def make_gym_env(max_episode_steps,env_name,verbose = False):
    gym_env = TimeLimit(gym.make(env_name),max_episode_steps=max_episode_steps)
    if verbose:
        print(f'for {env_name}, the low action is {gym_env.action_space.low} and hight is {gym_env.action_space.high}')
    return gym_env


def synchronized_train_multi(cfg):
    # init 
    
    cem_erl = CemERl(cfg)
    logger = instantiate_class(cfg.logger)

    n_processes = min(cfg.algorithm.num_processes,cfg.algorithm.es_algorithm.pop_size)
    pop_size = cfg.algorithm.es_algorithm.pop_size
    tmp_steps = 0

    acquisition_agents = []
    acquisition_actors = []

    for i in range(n_processes): 
        env_agent = NoAutoResetGymAgent(make_gym_env,{'max_episode_steps':cfg.env.max_episode_steps,
                                            'env_name':cfg.env.env_name},
                                            n_envs=cfg.algorithm.n_envs).to(cfg.algorithm.es_algorithm.device)
        action_agent = cem_erl.get_acquisition_actor(i).to(cfg.algorithm.es_algorithm.device)
        acquisition_actors.append(action_agent)
        temporal_agent = TemporalAgent(Agents(env_agent, action_agent))
        temporal_agent.seed(cfg.algorithm.env_seed)
        agent = AsynchronousAgent(temporal_agent)
        acquisition_agents.append(agent)

    n_interactions = 0

    rl_active = cem_erl.rl_active

    for epoch in tqdm.tqdm(range(cfg.algorithm.max_epochs)):
        if(n_interactions > cfg.algorithm.max_steps):
            break
        timing = time()
        acquisition_workspaces = []
        nb_agent_finished = 0
        while(nb_agent_finished < pop_size):
            n_to_launch = min(pop_size-nb_agent_finished, n_processes)
            for idx_agent in range(n_to_launch):        
                idx_weight = idx_agent + nb_agent_finished
                cem_erl.update_acquisition_actor(acquisition_actors[idx_agent],idx_weight)
                # TODO: add noise args to agents interaction with env ? Alois does not. 
                acquisition_agents[idx_agent](t=0,stop_variable="env/done")

            #Wait for agents execution
            running=True
            while running:
                are_running = [a.is_running() for a in acquisition_agents[:n_to_launch]]
                running = any(are_running)

            nb_agent_finished += n_to_launch
            acquisition_workspaces += [a.get_workspace() for a in acquisition_agents[:n_to_launch]]
        ## Logging rewards:
        for acquisition_worspace in acquisition_workspaces:
            n_interactions += acquisition_worspace.time_size() - 1
            

        agents_creward = torch.zeros(len(acquisition_workspaces))
        for i,acquisition_worspace in enumerate(acquisition_workspaces):
            done = acquisition_worspace['env/done']
            cumulated_reward = acquisition_worspace['env/cumulated_reward']
            creward = cumulated_reward[done]
            agents_creward[i] = creward.mean()
        
        print(f"/nTemps execution CEM {time() - timing}/n")

        logger.add_scalar(f"monitor/n_interactions", n_interactions, epoch)
        logger.add_scalar(f"monitor/reward", agents_creward.mean().item(), n_interactions)
        logger.add_scalar(f"monitor/reward_best", agents_creward.max().item(), n_interactions)

        agents_creward_sorted, indices = agents_creward.data.sort()
        elites = agents_creward_sorted.data[pop_size - cfg.algorithm.es_algorithm.elites_nb:pop_size]        
        logger.add_scalar(f"monitor/elites_reward", elites.mean().item(), n_interactions)
        
        if 0 in indices[pop_size - cfg.algorithm.es_algorithm.elites_nb:pop_size] and epoch % cfg.algorithm.es_algorithm.steps_es == 0:
            logger.add_scalar(f"monitor/rl_learner_selection", 1, n_interactions)
        else: 
            logger.add_scalar(f"monitor/rl_learner_selection", 0, n_interactions)
        
        timing = time()

        if(n_interactions - tmp_steps > cfg.data.logger_interval):
            tmp_steps += cfg.data.logger_interval
            with open(os.path.join(os.getcwd(),cfg.data.path), "a+", newline='') as f:
                writer = csv.writer(f)
                # csv file : steps,  reward, best reward, reward elite
                writer.writerow([tmp_steps, 
                                agents_creward.mean().item(),
                                agents_creward.max().item(),
                                elites.mean().item()])
            

        if rl_active :
            cem_erl.rl_activation = epoch % cfg.algorithm.es_algorithm.steps_es == 0

        cem_erl.train(acquisition_workspaces, agents_creward ,n_interactions,logger)
        
        print(f"/nTemps execution TD3 {time() - timing}/n")

    for a in acquisition_agents:
        a.close()


@hydra.main(config_path=os.path.join(os.getcwd(),'run_launcher/configs/'), config_name="cem_erl.yaml")
def main(cfg : DictConfig):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    synchronized_train_multi(cfg)
    # debug_train(cfg)

if __name__=='__main__':
    main()
