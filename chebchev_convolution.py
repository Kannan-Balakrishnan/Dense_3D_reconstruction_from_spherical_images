import torch
import torchvision
import numpy as np
import yaml

from small_encoder import SmallEncoder
from utils import utils

torch.manual_seed(1234)

data=np.load("./Dense_3D_reconstruction_from_spherical_images/processed_images/00000001.npy")
data=torchvision.transforms.functional.to_tensor(data)

class chebchevconvolution(torch.nn.Module):

    def __init__(self):
        super(chebchevconvolution, self).__init__()
        obj = utils()
        self.model_params = obj.model_parameters()
        self.create_architecture()
        
    def create_architecture(self)-> None:
        self.fnet = SmallEncoder(self.model_params)        
        self.cnet = SmallEncoder(self.model_params)
        

    def forward(self, x):
        h=self.fnet(x, self.model_params)
        
        #self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        return h
model = chebchevconvolution()
print(model)
h = model(data)
