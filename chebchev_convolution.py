import torch
from torch.nn import Linear
from torch_geometric.nn import ChebConv
import torchvision
import pygsp
import networkx as nx 
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from torch_geometric.nn import knn_graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import yaml
from healpix_pool_unpool import Healpix
import math

from small_encoder import SmallEncoder

torch.manual_seed(1234)

data=np.load("processed_images/00000001.npy")
data=torchvision.transforms.functional.to_tensor(data)
data.shape

class chebchevconvolution(torch.nn.Module):
    def __init__(self):
        super(chebchevconvolution, self).__init__()
        self.edge_index=[]
        self.weight=[]
        self.conv_layer=[]
        self.model_parameters()
        self.create_architecture()
        
    def create_architecture(self):
        self.fnet = SmallEncoder(self)        
        self.cnet = SmallEncoder(self)
        for i in range(self.depth):
            subdivisions = int(self.healpix_resolution_calculator(self.n_pixels)/2**i)
            Graph_pygsp = SphereHealpix(subdivisions, nest=True, k=20)
            graph_netx=Graph_pygsp.to_networkx()
            graph_pyg=from_networkx(graph_netx)
            edge_idx=graph_pyg.edge_index
            self.edge_index.append(edge_idx)
            weight=graph_pyg.weight
            self.weight.append(weight)
        
            
            
    def healpix_resolution_calculator(self,nodes):
        """Calculate the resolution of a healpix graph
        for a given number of nodes.
        Args:
            nodes (int): number of nodes in healpix sampling
        Returns:
            int: resolution for the matching healpix graph
        """ 
        resolution = int(math.sqrt(nodes / 12))
        return resolution
        
    def model_parameters(self):
        filename="config.yml"
        output_channels_convolutionlayer=[]
        with open(filename) as f:
            my_dict = yaml.safe_load(f)
            self.n_pixels=my_dict["IMAGE PARAMS"]["n_pixels"]
            self.norm_fn='batch'
            self.depth=my_dict["IMAGE PARAMS"]["depth"]
            self.conv1_input_channels=my_dict["MODEL PARAMS"]["conv1_input_channels"]
            self.conv1_output_channels=my_dict["MODEL PARAMS"]["conv1_output_channels"]
            self.conv1_kernel_size=my_dict["MODEL PARAMS"]["conv1_kernel_size"]
            self.bottleneck_layer1_input_channels=my_dict["MODEL PARAMS"]["bottleneck_layer1_input_channels"]
            self.bottleneck_layer1_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer1_kernel_size"]
            self.bottleneck_layer1_output_channels=my_dict["MODEL PARAMS"]["bottleneck_layer1_output_channels"]
            self.bottleneck_layer2_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer2_kernel_size"]
            self.bottleneck_layer2_output_channels=my_dict["MODEL PARAMS"]["bottleneck_layer2_output_channels"]
            self.bottleneck_layer3_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer3_kernel_size"]
            self.conv2_input_channels=my_dict["MODEL PARAMS"]["conv2_input_channels"]
            self.conv2_kernel_size=my_dict["MODEL PARAMS"]["conv2_kernel_size"]
            self.conv2_output_channels=my_dict["MODEL PARAMS"]["conv2_output_channels"]
            self.norm_fn=my_dict["MODEL PARAMS"]["norm_fn"]
            #assert (self.depth==len(self.output_channels_convolutionlayer)),"Please check the config.yml file. The depth (under image params) and len(output channel) (under Model params) should be equal"

        

    def forward(self, x):
        h=self.fnet(x, self)
        
        #self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        return self.h
model = chebchevconvolution()
print(model)
h = model(data)