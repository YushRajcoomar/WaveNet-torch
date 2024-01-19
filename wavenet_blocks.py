import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,dilation,*args,**kwargs):
    super(CausalConv1d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.pad = (self.kernel_size - 1) * self.dilation
    self.conv1d = nn.Conv1d(self.in_channels,self.out_channels,kernel_size=self.kernel_size,padding=self.pad,dilation=self.dilation,bias=False)

  def forward(self,x):
    # x should be in shape (batch_size,num_samples,timestamp)
    return self.conv1d(x)[:,:,:-self.pad]

class CausalConvBlock(nn.Module):
  def __init__(self,layers,kernel_size):
    super(CausalConvBlock, self).__init__()
    self.layers = layers # list form
    self.kernel_size = kernel_size
    self.module_list = nn.ModuleList()

    for i in range(len(layers)-1):
      in_channel,out_channel = layers[i], layers[i+1]
      self.module_list.append(CausalConv1d(in_channel,out_channel,kernel_size,dilation=1)) #maybe batchnorm here?

  def forward(self,x):
    for layer in self.module_list:
        x = layer(x)
    return x

class DilatedConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,dilation):
    super(DilatedConvBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.dilatedconv1d=CausalConv1d(in_channels,out_channels,kernel_size,dilation=dilation)

  def forward(self,x):
    return self.dilatedconv1d(x)

class ConvGate:
  def apply(x):
    tanh_x = torch.tanh(x)
    sigmoid_x = torch.sigmoid(x)
    return tanh_x * sigmoid_x

class DilatedResidualLayer(nn.Module):
  def __init__(self,in_channels,skip_channels,kernel_size,dilation):
    super(DilatedResidualLayer, self).__init__()
    self.dilated_conv_block = DilatedConvBlock(in_channels,in_channels,kernel_size,dilation)
    self.pointwise_conv_residual = nn.Conv1d(in_channels,in_channels,kernel_size=1,bias=True)
    self.pointwise_conv_skip = nn.Conv1d(in_channels,in_channels,kernel_size=1,bias=True)

  def forward(self,x):
    dilated_convolved_input = self.dilated_conv_block.forward(x)
    gated_input = ConvGate.apply(dilated_convolved_input)
    pre_skip = self.pointwise_conv_residual(gated_input)
    residual_input = pre_skip + x
    skip_result = self.pointwise_conv_skip(gated_input)
    return residual_input, skip_result

class DilatedResidualBlock(nn.Module):
  def __init__(self,num_layers,num_stack,in_channels,skip_channels,kernel_size):
    super(DilatedResidualBlock, self).__init__()
    self.num_layers = num_layers
    self.num_stack = num_stack
    self.in_channels = in_channels
    self.skip_channels = skip_channels
    self.kernel_size = kernel_size

  def get_dilations(self):
    dilations = []
    for l in range(self.num_stack):
      for i in range(self.num_layers):
        dilations.append(2**i)
    return dilations

  def forward(self,x):
    skip_results = []
    dilations = self.get_dilations()
    for dilation in dilations:
      dilated_residual_layer = DilatedResidualLayer(self.in_channels,self.skip_channels,self.kernel_size,dilation)
      x, skip_result = dilated_residual_layer.forward(x)
      skip_results.append(skip_result)
    return torch.stack(skip_results)

class DenseBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(DenseBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

  def quantized_softmax(self,x,mu):
      prob = torch.log(1 + (mu*torch.abs(x)))/torch.log(1+mu)
      return torch.sign(x) * prob

  def forward(self,x):
    x = F.relu(x)
    x = nn.Conv1d(self.in_channels,self.in_channels//5,kernel_size=1,bias=True)(x)
    x = F.relu(x)
    x = nn.Conv1d(self.in_channels//5,self.out_channels,kernel_size=1,bias=True)(x)
    x = F.relu(x)
    x = nn.Conv1d(self.out_channels,self.out_channels,kernel_size=1,bias=True)(x)
    output = F.softmax(x,dim=2)
    return output

class OutputBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,residual_blocks):
    super(OutputBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.residual_blocks = residual_blocks
    self.skip_channels = self.residual_blocks * 10
    self.dense_block = DenseBlock(self.in_channels,self.out_channels,self.kernel_size)

 
  def forward(self,x):
    output = self.dense_block.forward(x)
    return output