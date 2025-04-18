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
    update_target_every : int = 5 
    min_reward : int = -200 
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
    
    def render(self):
        pass