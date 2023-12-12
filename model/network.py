import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .P2IKT import P2IKT_model
import numpy as np
import torch.nn.functional as F



def build_net(model_name,data_mode,training=True,image_size=384):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg
            
    if model_name == 'P2IKT':
        return P2IKT_model()
