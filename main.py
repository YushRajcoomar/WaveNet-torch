import torch
import torch.nn as nn

from wavenet_blocks import *

x = torch.randn((1,1,2056))

causal_conv_block = CausalConvBlock(10,1,1,4)
x = causal_conv_block.forward(x)