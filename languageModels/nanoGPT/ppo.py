# Fine-Tuning with PPO for NanoGPT 
# This file contains the code for the PPO algorithm 
# It is based on the paper : https://arxiv.org/abs/2104.05625
# and the implementation : https://github.com/schinger/FullLLM/blob/main/ppo.py

import wandb
from dataclasses import dataclass, field
from typing import Literal, Optional, List
import tyro
import math
import os
import time
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
 
@dataclass 
class PPOConfig:
    """
    Configuration for the PPO algorithm 
    """
    pass
