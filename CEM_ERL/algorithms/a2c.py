import copy

from salina import Workspace,Agent,instantiate_class,get_class,get_arguments
from salina.agents import TemporalAgent,Agents
from torch.nn.utils import parameters_to_vector,vector_to_parameters

import torch

from algorithms.learner import learner
from utils import get_env_dimensions,soft_param_update

def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v

class A2C(learner):

    def __init__(self,cfg) -> None:

        self.cfg=cfg
        obs_dim,n_action = get_env_dimensions(self.cfg.env,discrete_action=True)
        # create agents
        common_nn_args = {'state_dim':obs_dim,'n_action':n_action}
        # To merge dict_a and dict_b in one line you can write :  {**dict_a,**dict_b}
        nn_args= {**dict(cfg.algorithm.prob_agent),**common_nn_args}
        self.prob_agent =  get_class(cfg.algorithm.prob_agent)(**nn_args)
        self.action_agent =  get_class(cfg.algorithm.action_agent)(**nn_args)

        nn_args= {**dict(cfg.algorithm.v_agent),**common_nn_args}
        self.v_agent = get_class(cfg.algorithm.v_agent)(**nn_args)

        # create temporal agents
        self.t_prob_agent = TemporalAgent(self.prob_agent)
        self.t_action_agent = TemporalAgent(self.action_agent)
        self.t_v_agent = TemporalAgent(self.v_agent)

        # create optimizers
        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        parameters = torch.nn.Sequential(self.prob_agent, self.v_agent).parameters()
        self.optimizer = get_class(cfg.algorithm.optimizer)(parameters,
                                            **optimizer_args)

    # def set_actor_params(self,weight):
    #     ''' Overrite the parameters of the actor and the target actor '''
    #     vector_to_parameters(weight,self.prob_agent.parameters())
    #     # reset action optimizer: 
    #     self.optimizer = torch.optim.Adam(self.prob_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)

    # def get_parameters(self):
    #     return self.self.prob_agent.parameters()
    
    def get_acquisition_actor(self):
        '''
        Returns the agents used to gather experiments in the environment.
        '''
        self.prob_agent.set_name("prob_agent")
        acquisition_action = Agents(self.prob_agent,self.action_agent)
        acquisition_action = copy.deepcopy(acquisition_action)
        return acquisition_action

    def get_acquisition_args(self):
        return {'stochastic':True}

    def updateAcquisitionAgent(self,acquisitionAgent):
        for a in acquisitionAgent.get_by_name("prob_agent"):
            a.load_state_dict(self.prob_agent.state_dict())        

    def train(self,acq_workspace,n_actor_steps,n_total_actor_steps,logger):
        replay_workspace = Workspace(acq_workspace)
        self.t_prob_agent(replay_workspace, t=0, n_steps=self.cfg.algorithm.n_timesteps)
        self.t_v_agent(replay_workspace, t=0, n_steps=self.cfg.algorithm.n_timesteps)
        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]
        target = reward[1:] + self.cfg.algorithm.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]
        td_error = td ** 2
        critic_loss = td_error.mean()

        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        logger.add_scalar("critic_loss", critic_loss.item(), n_total_actor_steps)
        logger.add_scalar("entropy_loss", entropy_loss.item(), n_total_actor_steps)
        logger.add_scalar("a2c_loss", a2c_loss.item(), n_total_actor_steps)
        loss = (    -self.cfg.algorithm.entropy_coef * entropy_loss
                    + self.cfg.algorithm.critic_coef * critic_loss
                    - self.cfg.algorithm.a2c_coef * a2c_loss
                )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
