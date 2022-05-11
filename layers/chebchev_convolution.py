import torch
import torchvision
import numpy as np
import yaml
from corr import CorrBlock

from small_encoder import SmallEncoder
from utils.utils import utils

torch.manual_seed(1234)

obj = utils()
model_params = obj.model_parameters()
image1=np.load(model_params.image_path + str(model_params.image1number).zfill(8) + ".npy")
image2=np.load(model_params.image_path + str(model_params.image2number).zfill(8) + ".npy")
image1 = 2 * (image1 / 255.0) - 1.0
image2 = 2 * (image2 / 255.0) - 1.0
image1=torchvision.transforms.functional.to_tensor(image1).float().contiguous()
image2=torchvision.transforms.functional.to_tensor(image2).float().contiguous()

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
        fmap1, fmap2 = self.fnet(x, self.model_params)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2)
        
        #self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        return h
model = chebchevconvolution()
print(model)
h = model([image1, image2])
