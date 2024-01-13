import torch
import torch.nn as nn
import torch.nn.functional as F

from wavenet_blocks import *

class WaveNetModel(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,residual_blocks):
    super(WaveNetModel, self).__init__()
    self.causal_conv1d_layers = 5
    self.dilated_block_layers = 10
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.residual_blocks = residual_blocks
    self.module_list = nn.ModuleList()

    self.causal_conv1d_block = CausalConvBlock(self.causal_conv1d_layers,self.in_channels,self.out_channels,self.kernel_size)

    for b in range(self.residual_blocks):
      self.module_list.append(DilatedResidualBlock(self.in_channels,self.out_channels,self.kernel_size))

    self.dense_block = OutputBlock(self.in_channels,self.out_channels,self.kernel_size,self.residual_blocks)


  def forward(self,x):
    skip_connections = []
    x = self.causal_conv1d_block.forward(x)
    for block in self.module_list:
      x = block.forward(x) #residual_signal
      skip_connections.append(x)
    skip_tensor = torch.cat(skip_connections,dim=1)

    output = self.dense_block.forward(x)
    return output