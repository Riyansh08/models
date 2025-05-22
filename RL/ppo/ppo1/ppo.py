import torch
import torch.nn as nn
import torch.nn.functional as F
import math , numpy as np  , random  
from typing import List , Tuple , Dict , Optional 
from dataclasses import dataclass , field
from torch.distributions import MultivariateNormal , Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

#SETTING DEVICE 
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print("USING A GPU ")
else:
    print("USING A CPU ")
#PPO POLICY 
class RoloutBuffer():
    def __init__(self):
        self.actions = []
        self.states= []
        self.logprobs =[]
        self.rewards =[]
        self.is_terminals = []
        self.states_values =[]
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.states_values[:]

class ActorCritic(nn.Module):
    def __init__(self , state_dim , action_dim , continous_action , action_std_init):
        super(ActorCritic , self).__init__()
        self.continous_action = continous_action 
        # STD = summation (x-mean)**2 / N-1 the square root 
        if continous_action:
            self.action_dim = action_dim 
            self.action_std = action_std_init 
            self.action_var = torch.full((action_dim ,) , action_std_init * action_std_init)
        #ACTOR - produces prob or distribution of actions 
        #CRITIC - produces value function 
        # Used for calculating ADvantage function 
        
        if continous_action:
            self.actor = nn.Sequential(
                nn.Linear(state_dim , 64), 
                nn.Tanh(),  #Normalize the output to be between -1 and 1
                nn.Linear(64 , 64), 
                nn.Tanh() , 
                nn.Linear(64 , action_dim) , 
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim , 64), 
                nn.Tanh(),
                nn.Linear(64 , 64), 
                nn.Tanh(), 
                nn.Linear(64 , action_dim),
                nn.Softmax(dim = -1)
            )
            
        #CRITIC
        #Valu function - estimate of the state value 
        self.critic = nn.Sequential(
            nn.Linear(state_dim , 64), 
            nn.Tanh(),
            nn.Linear(64 , 64), 
            nn.Tanh(), 
            nn.Linear(64 , 1)
        )
            
    def set_action_std(self, new_action_std):
        if self.continous_action:
            self.action_vat = torch.full((self.action_dim ,) , new_action_std * new_action_std)
        else: 
            print("CAN'T SET ACTION STD FOR CATEGORICAL ACTIONS")
            
    def forward(self):
        raise NotImplementedError
    
    def act(self , state):
        if self.continous_action:
            action_mean = self.actor(state) 
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean , cov_mat)
        else: 
            action_prob = self.actor(state)
            dist  = Categorical(action_prob)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action.detach() , action_logprob.detach() , state_val.detach()
    # Evaluate the action 
    # DURING TRAINING 
    def evaluate(self , state , action):
        if self.continous_action:
            action_mean = self.actor(state)
            action_var  = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean , cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs= dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_val  = self.critic(state)
        
        return action_logprobs , state_val , dist_entropy
    
class PPO(nn.Module):
    def __init__(self , state_dim , action_dim , continous_action , lr_actor , lr_critic , gamma , K_epochs , eps_clip , action_std_init = 0.6 ):
        super(PPO, self).__init__()
        self.continous_action = continous_action 
        if continous_action:
            self.action_std = action_std_init 
        self.gamma = gamma 
        self.K_epochs = K_epochs 
        self.eps_clip = eps_clip 
        
        self.buffer = RoloutBuffer()
        
        self.policy = ActorCritic(state_dim , action_dim , continous_action , action_std_init)
        self.optimizer = torch.optim.AdamW([
            {"params": self.policy.actor.parameters() , "lr": lr_actor},
            {"params": self.policy.critic.parameters() , "lr": lr_critic}
        ])       
        
        self.policy_old = ActorCritic(state_dim , action_dim , continous_action , action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        if self.continous_action:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("CAN'T SET ACTION STD FOR CATEGORICAL ACTIONS")

    def decay_action_std(self , action_std_decay_rate , min_action_std):
        if self.continous_action:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std , 4)
            if self.action_std <= min_action_std:
               self.action_std = min_action_std
               print("Setting action std to min value : ", self.action_std)
        else:
            print("CAN'T DECAY ACTION STD FOR CATEGORICAL ACTIONS")
    # IMPLEMENTING THE CRUX OF PPO       
    def select_action(self , state):
        if self.continous_action:
            with torch.no_grad():
                state  = torch.FloatTensor(state)
                action , action_logprob , state_val = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            return action.item()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                action, action_logprob , state_val = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            return action.item()
    def update(self):
        rewards = []
        discounted_reward = 0 
        for reward , is_terminal in zip(reversed(self.buffer.rewards) , reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0 
            discounted_reward = reward + self.gamma * discounted_reward 
            rewards.insert(0 , discounted_reward)
        
        rewards  = torch.tensor(rewards , dtype = torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(self.buffer.states) , dim=0).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions) ,dim = 0 ).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs) , dim = 0).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values) , dim = 0).detach()
        
        advantages = rewards.detach() - old_state_values.detach()
        
        for _ in range(self.K_epochs):
            logprobs , state_values , dist_entropy = self.policy.evaluate(old_states , old_actions)
            
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss 
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios , 1-self.eps_clip , 1+self.eps_clip) * advantages 
            
            loss = -torch.min(surr1 , surr2) + 0.5 * self.MseLoss(state_values , rewards) - 0.01* dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))