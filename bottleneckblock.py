import torch
import torch_geometric.nn as nn
from utils import utils

class BottleneckBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, params, itr, stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.ChebConv(in_planes, planes//4, params.bottleneck_layer1_kernel_size)
        self.conv2 = nn.ChebConv(planes//4, planes//4, params.bottleneck_layer2_kernel_size)
        self.conv3 = nn.ChebConv(planes//4, planes,params.bottleneck_layer3_kernel_size)
        self.relu = torch.nn.ReLU(inplace=True)

        obj = utils()
        
        subdivisions = int(obj.healpix_resolution_calculator(params.n_pixels)/2**itr)
        self.edge_index,self.edge_weight = obj.create_graph_func(subdivisions)
        
        num_groups = planes // 8
        
        if params.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm(planes//4)
            self.norm2 = nn.BatchNorm(planes//4)
            self.norm3 = nn.BatchNorm(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm(planes)
        
        elif params.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm(planes//4)
            self.norm2 = nn.InstanceNorm(planes//4)
            self.norm3 = nn.InstanceNorm(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm(planes)

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = torch.nn.Sequential(nn.ChebConv(in_planes,
                  planes, 1), self.norm4)


    def forward(self, x):
        y= x
        print ("x",x.shape)
        print ("y",y.shape)
        y = self.conv1(y, self.edge_index, self.edge_weight)
        print(self.edge_index.shape)
        y = y.transpose(1, 2)
        y = self.relu(self.norm1(y))
        y = y.transpose(2, 1)
        y = self.conv2(y, self.edge_index, self.edge_weight)
        y = y.transpose(1, 2)
        y = self.relu(self.norm2(y))
        y = y.transpose(2, 1)
        y = self.conv3(y, self.edge_index, self.edge_weight)
        y = y.transpose(1, 2)
        y = self.relu(self.norm3(y))
        y = y.transpose(2, 1)
        print ("Y after convolution",y.shape)
        if self.downsample is not None:
           x = self.downsample(x)
        print ("X after convolution",x.shape)
        return self.relu(x+y)

#params=[]
#params.bottleneck_layer1_kernel_size=1
#params.bottleneck_layer2_kernel_size=3
#params.bottleneck_layer3_kernel_size=1
#model=BottleneckBlock(3,)
