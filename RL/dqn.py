#Deep Q Learning from scratch playing the game of blob
#reference - https://github.com/cneuralnetwork/solving-ml-papers/blob/main/DQN/Deep%20Q%20Networks.py
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
    
# Environment define
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
        pass
    def move(self , x = False , y = False):
        pass
