#Deep Q Learning from scratch playing the game of blob
#reference/ credit - https://github.com/cneuralnetwork/solving-ml-papers/blob/main/DQN/Deep%20Q%20Networks.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hope , dedication  # type: ignore
import torch.optim as optim 
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter
from collections import deque 
import time
import random 
from typing import List
from tqdm import tqdm 
import os 
import cv2 # type: ignore
from dataclasses import dataclass , field 

@dataclass
class Config:
    # Model configuration
    model_name : str = "DQN"
    min_replay_memory_size : int = 1000 
    minibatch_size : int = 64
    update_target_every : int = 10 
    min_reward : int = -200 
    max_buffer_size : int = 50000
    memory_fraction : float = 0.2
    epilson : int = 1 #starting with full exploration.then gradually decaying
    eposides : int = 20000
    decay_rate : float = 0.99997
    min_epilson : float = 0.05
    aggreagate_stats_every : int = 50
    show_preview : bool =False
    
# BLOB game define
class Blob:
    def __init__(self  , size:int , color:tuple):
        self.size = size 
        self.color = color
        # spawn random location
        self.x = np.random.randint(0 , size)
        self.y = np.random.randint(0 , size) 
    def __str__(self):
        return f"Blob(size={self.size}, color={self.color}, x={self.x}, y={self.y})"
    def __sub__(self , other):
        return (self.x-other.x , self.y-other.y)
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)
    def move(self , x = False , y = False):
        if x:
            self.x += x
        else:
            self.x += np.random.randint(-1, 2)
        if y:
            self.y += y
        else:
            self.y += np.random.randint(-1, 2)
        # check for out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > 100:
            self.x = 100
        if self.y < 0:
            self.y = 0
        elif self.y > 100:
            self.y = 100
#Blob environment define 
class BlobEnv:
    SIZE = 10 
    RETURN_IMAGES = True
    MOVE_PENALTY = 1 
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25 
    OBSERVATION_STATE_VALUE = (SIZE , SIZE , 3)
    ACTION_SPACE_SIZE = 9 
    PLAYER_N = 1 
    FOOD_N = 2 
    ENEMY_N = 3 
    d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    
    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food  = Blob(self.size)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)
        self.episode_step = 0 
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = np.array([self.player - self.food , self.player - self.enemy])
        return observation 
    def step(self , action):
        self.episode_step+=1
        self.player.action(action)
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = np.array([self.player - self.food , self.player - self.enemy])
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY 
            # reward -= self.ENEMY_PENALTY
        elif self.player == self.food:
            reward=  self.FOOD_REWARD
        else:
            reward = - self.MOVE_PENALTY
        done = False
        if reward == self.FOOD_REWARD or reward == self.ENEMY_PENALTY or self.episode_step >=200:
            done =True 
        return new_observation , reward , done
    
    def get_image(self):
        env = np.zeros((self.SIZE , self.SIZE , 3) , dtype=np.uint8)
        env[self.food.y , self.food.x] = self.d[self.FOOD_N]
        env[self.enemy.y , self.enemy.x] = self.d[self.ENEMY_N]
        env[self.player.y , self.player.x] = self.d[self.PLAYER_N]
        img = Image.fromarray(env , 'RGB')
        return img
    def render(self ):
        img = self.get_image()
        img = img.resize( (300 , 300) )
        cv2.imshow("Blob Env" , np.array(img))
        cv2.waitKey(1)
        
#DEFINING ENVIRONMENT
env = BlobEnv()
ep_rewards = [-200]
random.seed(8)
np.random.seed(8)
tf.random.set_seed(8)
model = Blob()
#MODEL PATH     
if not os.path.isdir('models'):
    os.mkdir('models')
# CREATING THE DQN AGENT 
class DQNAGENT(nn.Module):
    def __init__(self , config: Config):
        super(DQNAGENT , self).__init__()
        self.config = config
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen = self.config.max_buffer_size)
        self.target_update = self.config.update_target_every
        self.min_replay_memory_size = self.config.min_replay_memory_size 
        self.memory_fraction = self.config.memory_fraction
        self.min_epilson = self.config.min_epilson
        self.episodes = self.config.episodes
        self.decay_rate = self.config.decay_rate
        self.aggreagate_stats_every = self.config.aggreagate_stats_every
        self.show_preview = self.config.show_preview
        self.device = torch.cuda if torch.cuda.is_available() else torch.cpu
    def create_model(self):
        #NORMAL CNN MODEL 
        model = nn.Sequential()
        model.add(Conv2d(256 , (3,3) , input_shape = env.OBSERVATION_STATE_VALUE))
        model.add(Activation('relu'))
        model.add(Conv2d(128 , (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2d((2,2)))
        model.add(Conv2d(64 , (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2d((2,2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(optimizer = ADAM(learning_rate = 0.001) , loss = 'mse')
        return model
    def update_replay_memory(self , new_transition):
        self.replay_memory.append(new_transition)
    def get_qs(self , state):
        self.model.predict(np.array(state).reshape(-1 , *state.shape)/ 255)[0]
    def train(self , terminal_state ,  train_episodes = 100000 ):
        if self.replay_memory < self.min_replay_memory_size:
            return 
        minibatch = random.sample(self.replay_memory , self.config.minibatch_size) # Sampling for training randomly to avoid similar patterns in close episodes 
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        new_states = np.array([transition[3] for transition in minibatch])/ 255 
        future_qs = self.target_model.predict(new_states)
        X = []
        Y = []
        for index , (current_state , action , reward , new_state , done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.config.discount * max_future_q 
            else:
                new_q = reward 
            current_qs = current_states[index]
            current_qs[action] = new_q
            X.append(current_state)
            Y.append(current_qs)
            self.model.fit(
            np.array(X)/255.0,
            np.array(y),
            batch_size=self.config.minibatch_size,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None
        )
        # update target network
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > self.config.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
agent = DQNAGENT(Config())

for episode in tqdm(range(1, config.episodes+1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    curr_state = env.reset()
    done = False

    while not done:
        if np.random.random() > agent.epsilon:
            action = np.argmax(agent.get_qs(curr_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        new_state, reward, done = env.step(action)
        episode_reward += reward
        if config.show_preview and episode % config.aggregate_stats_every == 0:
            env.render()
        agent.update_replay_memory((curr_state, action, reward, new_state, done))
        agent.train(done, step)
        curr_state = new_state
        step += 1
    ep_rewards.append(episode_reward)
    if episode % config.aggregate_stats_every == 0 or episode == 1:
        avg_reward = sum(ep_rewards[-config.aggregate_stats_every:]) / config.aggregate_stats_every
        min_reward = min(ep_rewards[-config.aggregate_stats_every:])
        max_reward = max(ep_rewards[-config.aggregate_stats_every:])
        agent.tensorboard.update_stats(
            reward_avg=avg_reward,
            reward_min=min_reward,
            reward_max=max_reward,
            epsilon=agent.epsilon
        )
        if min_reward >= config.min_reward:
            agent.model.save(
                f"models/{config.model_name}__"
                f"{max_reward:_>7.2f}max_"
                f"{avg_reward:_>7.2f}avg_"
                f"{min_reward:_>7.2f}min__"
                f"{int(time.time())}.model"
            )