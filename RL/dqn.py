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
    pass