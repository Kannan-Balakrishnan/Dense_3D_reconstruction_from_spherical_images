import torch
import torch.nn as torchnn
import torch_geometric.nn as nn

from bottleneckblock import BottleneckBlock
from healpix_pool_unpool import Healpix
class SmallEncoder(torch.nn.Module):
    def __init__(self,params):
        super(SmallEncoder, self).__init__()
        #self.norm_fn = norm_fn

        if params.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm(num_groups=8, num_channels=32)
            
        elif params.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm(32)

        #elif self.norm_fn == 'instance':
         #   self.norm1 = nn.InstanceNorm2d(32)

        #elif self.norm_fn == 'none':
          #  self.norm1 = nn.Sequential()

        self.conv1 = nn.ChebConv(params.conv1_input_channels, params.conv1_output_channels, params.conv1_kernel_size)
        self.relu1 = torchnn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(params,params.bottleneck_layer1_input_channels,  stride=1)
        self.layer2 = self._make_layer(params,params.bottleneck_layer1_output_channels, stride=2)
        self.layer3 = self._make_layer(params,params.bottleneck_layer2_output_channels, stride=2)

        self.dropout = 0
        if self.dropout > 0:
            self.dropout = nn.Dropout2d(p=self.dropout)
        
        self.conv2 = nn.ChebConv(params.conv2_input_channels, params.conv2_output_channels, params.conv2_kernel_size)

        self.pool =Healpix()
            

    def _make_layer(self,params, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, params)
        layer2 = BottleneckBlock(dim, dim, params)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return mySequential(*layers)


    def forward(self, x, params):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x, params.edge_index[0], params.weight[0])
        x = self.norm1(x.transpose(1, 2))
        x = self.relu1(x)
        x = x.transpose(1, 2)

        x = self.layer1((x,params.edge_index[0],params.weight[0]))
        x,_,_ = x[0]
        x = self.pool.pooling(x)
        x = self.layer2((x,params.edge_index[1],params.weight[1]))
        x,_,_ = x[0]
        x = self.pool.pooling(x)
        x = self.layer3((x,params.edge_index[2],params.weight[2]))
        x,_,_ = x[0]
        x = self.pool.pooling(x)
        x = self.conv2(x,params.edge_index[2],params.weight[2])

        #if self.training and self.dropout is not None:
        #    x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

class mySequential(torch.nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
                input = module(*input)
        return input