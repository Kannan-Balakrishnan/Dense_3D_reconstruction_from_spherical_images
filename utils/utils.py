from numpy import array
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from torch_geometric.utils import from_networkx
import math
import yaml

class utils:
    def __init__(self) -> None:
        pass

    def create_graph_func(self, subdivisions)-> tuple:
        graph_pyg = from_networkx(SphereHealpix(subdivisions, nest=True, k=20).to_networkx()) 
        return (graph_pyg.edge_index, graph_pyg.weight)

    @staticmethod        
    def healpix_resolution_calculator(nodes)-> int:
        """Calculate the resolution of a healpix graph
        for a given number of nodes.
        Args:
            nodes (int): number of nodes in healpix sampling
        Returns:
            int: resolution for the matching healpix graph
        """ 
        resolution = int(math.sqrt(nodes / 12))
        return resolution

    def model_parameters(self)-> None:
        filename="./Dense_3D_reconstruction_from_spherical_images/config.yml"
        output_channels_convolutionlayer=[]
        with open(filename) as f:
            my_dict = yaml.safe_load(f)
            self.n_pixels=my_dict["IMAGE PARAMS"]["n_pixels"]
            self.norm_fn='batch'
            self.depth=my_dict["IMAGE PARAMS"]["depth"]
            self.conv1_input_channels=my_dict["MODEL PARAMS"]["conv1_input_channels"]
            self.conv1_output_channels=my_dict["MODEL PARAMS"]["conv1_output_channels"]
            self.conv1_kernel_size=my_dict["MODEL PARAMS"]["conv1_kernel_size"]
            self.bottleneck_layer_input_channels=my_dict["MODEL PARAMS"]["bottleneck_layer_input_channels"]
            self.bottleneck_layer1_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer1_kernel_size"]
            self.bottleneck_layer_output_channels=my_dict["MODEL PARAMS"]["bottleneck_layer_output_channels"]
            self.bottleneck_layer2_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer2_kernel_size"]
            self.bottleneck_layer2_output_channels=my_dict["MODEL PARAMS"]["bottleneck_layer2_output_channels"]
            self.bottleneck_layer3_kernel_size=my_dict["MODEL PARAMS"]["bottleneck_layer3_kernel_size"]
            self.bottleneck_layer_stride = my_dict["MODEL PARAMS"]["bottleneck_layer_stride"]
            self.conv2_input_channels=my_dict["MODEL PARAMS"]["conv2_input_channels"]
            self.conv2_kernel_size=my_dict["MODEL PARAMS"]["conv2_kernel_size"]
            self.conv2_output_channels=my_dict["MODEL PARAMS"]["conv2_output_channels"]
            self.norm_fn=my_dict["MODEL PARAMS"]["norm_fn"]
            assert (self.depth==len(self.bottleneck_layer_input_channels)),"Please \
                    check the config.yml file. The depth (under image params) and len(bottleneck_layer_input_channels) \
                    (under Model params) should be equal"

            return self
