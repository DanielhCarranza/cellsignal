"""Adapted Implementation of ConvNext for 8-bit x 6 channel images. """

from typing import Dict, Any
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.model.layers import trunc_normal_, DropPath
from timm.model.registry import register_model



