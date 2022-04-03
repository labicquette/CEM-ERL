import torch
from salina import instantiate_class,get_class
import random 
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from salina import Agent
import tqdm
import copy

from time import time

class CemERl:

    def __init__(self,cfg) -> None:

        # debug hyper-parameters : 
        self.rl_active = cfg.algorithm.rl_algorithm.active
        self.es_active = cfg.algorithm.es_algorithm.active

        # hyper-parameters: 
        self.pop_size = cfg.algorithm.es_algorithm.pop_size
        self.initial_buffer_size = cfg.algorithm.initial_buffer_size
        self.n_rl_agent = cfg.algorithm.n_rl_agent

        # device
        self.es_device = cfg.algorithm.es_algorithm.device
        self.rl_device = cfg.algorithm.learner.device
        if self.es_device == 'cuda' or self.rl_device == 'cuda':
            assert torch.cuda.is_available(), 'Cuda is not available'

        # RL objects:
        self.rl_learner =  get_class(cfg.algorithm.rl_algorithm)(cfg)
        self.rl_activation = True

        # CEM objects
        actor_weights = self.rl_learner.get_acquisition_actor().parameters()
        self.centroid = copy.deepcopy(parameters_to_vector(actor_weights).detach().to(self.es_device))
        code_args = {'num_params': len(self.centroid),'mu_init':self.centroid}
        kwargs = {**cfg.algorithm.es_algorithm, **code_args}
        self.es_learner = get_class(cfg.algorithm.es_algorithm)(**kwargs)

        self.pop_weights = self.es_learner.ask(self.pop_size)

        # vector_to_parameters does not seem to work when module are in different processes
        # the transfert agent is used to transfert vector_to_parameters in main thread
        # and then transfert the parameters to another agent in another process.
        self.param_transfert_agent = copy.deepcopy(self.rl_learner.get_acquisition_actor())

    def get_acquisition_actor(self,i) -> Agent:
        actor = self.rl_learner.get_acquisition_actor()
        weight = self.pop_weights[i]

        vector_to_parameters(weight,self.param_transfert_agent.parameters())
        actor.load_state_dict(self.param_transfert_agent.state_dict())
        return actor

    def update_acquisition_actor(self,actor,i) -> None:
        weight = self.pop_weights[i]
        vector_to_parameters(weight,self.param_transfert_agent.parameters())        
        actor.load_state_dict(self.param_transfert_agent.state_dict())

    def train(self,acq_workspaces,n_total_actor_steps,logger) -> None:

        # Compute fitness of population
        n_actor_all_steps = 0
        fitness = torch.zeros(len(acq_workspaces))
        for i,workspace in enumerate(acq_workspaces):
            n_actor_all_steps += (
                workspace.time_size() - 1
            ) * workspace.batch_size()
            done = workspace['env/done']
            cumulated_reward = workspace['env/cumulated_reward']
            fitness[i] = cumulated_reward[done].mean()
        
        if self.es_active:
            self.es_learner.tell(self.pop_weights,fitness) #  Update CEM
            self.pop_weights = self.es_learner.ask(self.pop_size) # Generate new population

        # RL update : 
        if self.rl_active:
            self.rl_learner.workspace_to_replay_buffer(acq_workspaces)
            if self.rl_learner.replay_buffer.size() < self.initial_buffer_size: # shouldn't access directly to replay buffer 
                return
            if not self.rl_activation:
                return
            
            temps_workspace = 0
            temps_learner = 0
            temps_total1 = 0
            temps_total2 = 0
            for _ in tqdm.tqdm(range(n_actor_all_steps),desc="Critic"):
                n_grad =  n_total_actor_steps # TODO: change logging method. 
                
                temps = time()
                train_workspace =  self.rl_learner.replay_buffer.get(self.rl_learner.cfg.algorithm.batch_size)
                temps_workspace += time() - temps

                temps = time()
                temps1, temps2 = self.rl_learner.train_critic(train_workspace,n_grad,logger)
                temps_total1 += temps1
                temps_total2 += temps2
                temps_learner += time() - temps
            
            print("temps workspace", temps_workspace)
            print("temps learner", temps_learner)
            print("temps1",temps_total1)
            print("temps2",temps_total2)

            for _ in tqdm.tqdm(range(n_actor_all_steps//2),desc="Actor"):
                n_grad =  n_total_actor_steps
                train_workspace =  self.rl_learner.replay_buffer.get(self.rl_learner.cfg.algorithm.batch_size)
                self.rl_learner.train_actor(train_workspace,n_grad,logger)
                
            # send back the updated weight into the population
            vector_param = torch.nn.utils.parameters_to_vector(self.rl_learner.get_parameters())
            self.pop_weights[0] = vector_param.detach().to(self.es_device)
