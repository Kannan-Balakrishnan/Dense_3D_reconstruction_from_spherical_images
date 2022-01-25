import torch
import torch_geometric.nn as nn

from bottleneckblock import BottleneckBlock
from healpix_pool_unpool import Healpix
from utils import utils

class SmallEncoder(torch.nn.Module):
    def __init__(self,params):
        super(SmallEncoder, self).__init__()
        #self.norm_fn = norm_fn

        obj = utils()
        subdivisions = int(obj.healpix_resolution_calculator(params.n_pixels)/2**0)
        self.edge_index_first_conv_layer,self.edge_weight_first_conv_layer = obj.create_graph_func(subdivisions)
        subdivisions = int(obj.healpix_resolution_calculator(params.n_pixels)/2**params.depth)
        self.edge_index_last_conv_layer,self.edge_weight_last_conv_layer = obj.create_graph_func(subdivisions)

        self.pool = Healpix()

        if params.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm(num_groups=8, num_channels=32)
            
        elif params.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm(32)


        #elif self.norm_fn == 'instance':
         #   self.norm1 = nn.InstanceNorm2d(32)

        #elif self.norm_fn == 'none':
          #  self.norm1 = nn.Sequential()

        self.conv1 = nn.ChebConv(params.conv1_input_channels, params.conv1_output_channels, params.conv1_kernel_size)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.in_planes = 32
        self.bottlenecklayers = []
        for _ in range(params.depth):
            self.bottlenecklayers.append(self._make_layer(params, 
                    params.bottleneck_layer_input_channels[_],
                    params.bottleneck_layer_stride[_], itr=_))
            self.bottlenecklayers.append(self.pool.pooling)
        self.bottlenecklayers = torch.nn.Sequential(*self.bottlenecklayers)

        self.dropout = 0
        if self.dropout > 0:
            self.dropout = nn.Dropout2d(p=self.dropout)
        
        self.conv2 = nn.ChebConv(params.conv2_input_channels, params.conv2_output_channels, params.conv2_kernel_size)

        
            

    def _make_layer(self,params, dim, stride, itr=1):
        params.first_layer = 1
        layer1 = BottleneckBlock(self.in_planes, dim, params, itr,
                        stride=stride)
        params.first_layer = 0
        layer2 = BottleneckBlock(dim, dim, params, itr, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return torch.nn.Sequential(*layers)


    def forward(self, x, params):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x, self.edge_index_first_conv_layer, self.edge_weight_first_conv_layer)
        x = self.norm1(x.transpose(1, 2))
        x = self.relu1(x)
        x = x.transpose(1, 2)
        x = self.bottlenecklayers(x)
        x = self.conv2(x,self.edge_index_last_conv_layer, self.edge_weight_last_conv_layer)

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
