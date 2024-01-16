import torch
import torch.nn as nn
import torch.nn.functional as F

from wavenet_blocks import *

class WaveNetModel(nn.Module):
  def __init__(self,in_channels,out_channels,skip_channels,kernel_size,residual_blocks):
    super(WaveNetModel, self).__init__()
    self.causal_conv1d_layers = [in_channels//(2**i) for i in [0,1,2,1,0]] ### 128 -> 64 -> 32 -> 64 -> 128
    self.dilated_block_layers = 10
    self.dilated_blocks = 5
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.skip_channels = skip_channels
    self.kernel_size = kernel_size
    self.residual_blocks = residual_blocks
    self.module_list = nn.ModuleList()

    self.causal_conv1d_block = CausalConvBlock(self.causal_conv1d_layers,self.kernel_size)
    
    for b in range(self.residual_blocks):
      self.module_list.append(DilatedResidualBlock(self.in_channels,self.out_channels,self.kernel_size))

    self.dense_block = OutputBlock(self.skip_channels * self.dilated_blocks,self.out_channels,self.kernel_size,self.residual_blocks)


  def forward(self,x):
    skip_connections = []
    x = self.causal_conv1d_block.forward(x)
    for block in self.module_list:
      x, skip_connection = block.forward(x) #residual_signal
      skip_connections.append(skip_connection)
    skip_tensor = torch.cat(skip_connections,dim=1)

    output = self.dense_block.forward(skip_tensor) # skip_channels * num layers
    return output