import numpy as np
import cv2
import argparse
import time
import healpy as hp
import sys
import torch
import torchvision
import torch_geometric.nn as nn
from torch_geometric.utils import to_networkx
import networkx as nx
sys.path.append("..")

from utils.coordinateconversion import CoordinateConversions
from utils.healpix_sampling import calNpix
from utils.utils import utils
from epipolar_geometry.matlab_visualization import plot
from epipolar_geometry.load_image import load_image, loadExtrinsicFile


class epipolar_geometry():
    ##################################################################################################################
    #Define arguments
    def __init__(self) -> None:
        args = self.arguments_parser()
        args.filename0 = str(args.reference_image_number).zfill(8)
        args.filename1 = str(args.image1number).zfill(8)
        args.filename2 = str(args.image2number).zfill(8)
        args.coord_convert = CoordinateConversions()
        args.obj = utils()
        ipix, NPIX=calNpix(args.NSIDE)
        #Getting the values of phi and theta for all the pixels
        phi, theta = np.degrees(hp.pix2ang(nside=args.NSIDE, ipix=ipix)) #where phi is between 0 and 180, theta is 
                                                            #between 0 and 360 

        #Converting all the points in the image (healpix image) into unit cartesian coordinates
        args.unit_cartesian_vector = np.asanyarray(args.coord_convert.sphericalToCartesian(phi, theta, 1))

        subdivisions = int(args.obj.healpix_resolution_calculator(NPIX))
        args.edge_index, args.edge_weight = args.obj.create_graph_func(subdivisions, k=8)

        self.load_extrinsics(args)
        self.args = args

        #self.epipolar_line_cal(args,args.R12, args.t12)


    def arguments_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default="processed_images/", help="Path of the Image folder")
        parser.add_argument("--extrinsics_path", default="extrinsics/", help="Path of the extrinsics folder")
        parser.add_argument("--max_depth", default=30, type=int, help = "Maximum depth of the image")
        parser.add_argument("--min_depth", default=1, type=int, help = "Minimum depth of the image")
        parser.add_argument("--divisions", default=15000, type=int,
            help="Divide 360 degrees by number of divisions to get angle of rotation around normal axis")
        parser.add_argument("--reference_image_number", default="0",
            help="The name of the reference image (all Rotations and translations are with respect to the reference image")
        parser.add_argument("--image1number", default="5",
            help="The name of first image (give only the number eg 1 or 2)")
        parser.add_argument("--image2number", default="3",
            help="The name of first image (give only the number eg 1 or 2)")
        parser.add_argument("--resize", type=bool, default=True, help="resizes the image for faster computation")
        parser.add_argument("--NSIDE", type=int, default=2, help="Number of sides in healpix pixellation" )
        parser.add_argument("--epipolar_conv1_input_channels", type=int, default=3, help="No of input channels")
        parser.add_argument("--epipolar_conv1_output_channels", type=int, default=16, help="No of output channels")
        parser.add_argument("--conv1_kernel_size", type=int, default=5, help="Chebchev kernel size")
        args = parser.parse_args()

        return args


    #####################################################################################################################
    #Calculate epipolar line

    def epipolar_line_cal(self, args, R, t, fmap1, fmap2):
        t_normalized = (t / np.sqrt(np.dot(t, t))).reshape(1,3)  #Normalizing the t vector

        #Transforming the x_vector from Image 1 to Image 2
        x_prime_vector = np.matmul(R, (args.unit_cartesian_vector * args.min_depth)) + t[:, None]
        x_prime_vector = x_prime_vector / np.sqrt(np.sum(x_prime_vector * x_prime_vector, axis = 0))
        x_prime_vector_scaled = np.matmul(R, (args.unit_cartesian_vector * args.max_depth)) + t[:, None]
        x_prime_vector_scaled /= np.sqrt(np.sum(x_prime_vector_scaled * x_prime_vector_scaled, axis = 0))

        #Calculating the normal vector to define the plane
        x_prime_vector = x_prime_vector.T

        normal_vector = (np.cross(t_normalized[:, None, :], x_prime_vector[None, :, :])).reshape(-1,3)
        normal_vector = normal_vector / np.sqrt(np.sum(normal_vector * normal_vector, axis = 1)).reshape(-1,1)

        corr = []

        #Build a dot product of normal_vector and all the points in the second image
        for i in range(0, args.unit_cartesian_vector.shape[1]):  #len(normal_vector)
            #img1, img2 = load_image(args)
            coplanar_points_idx = abs(np.dot(args.unit_cartesian_vector.T, normal_vector[i]))<0.01
            coplanar_points = (args.unit_cartesian_vector.T)[coplanar_points_idx==1]
            # pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(unit_cartesian_vector.T[i], 
            #                         imgWidth, imgHeight))
            # cv2.circle(img1, (int(pixel[0]), int(pixel[1])), radius=10, color=(255, 0, 0), thickness=-1)
            # cv2.imshow("image1",img1)
            epipolar_line = coplanar_points[np.where(np.sum(\
                np.cross(coplanar_points, x_prime_vector[i].T)*np.cross(coplanar_points, x_prime_vector_scaled.T[i]), axis=1)<0)]
            #epipolar_line = coplanar_points[epipolar_line_idx]
            #epipolar_line = torch.from_numpy(epipolar_line[np.where(np.sum(\
            #    np.cross(x_prime_vector[i], x_prime_vector_scaled.T[i])*np.cross(epipolar_line, x_prime_vector_scaled.T[i]), axis=1)>0)])
            epipolar_line = epipolar_line[np.where(np.sum(\
                np.cross(x_prime_vector[i], x_prime_vector_scaled.T[i])*np.cross(epipolar_line, x_prime_vector_scaled.T[i]), axis=1)>0)]
            #epipolar_line = epipolar_line[epipolar_line_idx2]

            #unit_cartesian_vector_idx = np.empty([1])
            #edge_index_filtered = np.empty([2,1])
            #edge_weight_filtered = np.empty([1])
            ucv_indices = []
            for j in range(len(epipolar_line)):
                ucv_indices.append(np.where(args.unit_cartesian_vector[0]==epipolar_line[j][0]))
            feature_dot_product = torch.matmul(fmap1[0][i],fmap2[0][ucv_indices].T)
            corr.append(feature_dot_product)
                #edge_index_indices = args.edge_index.T[np.where(args.edge_index[0] == ucv_indices[0][0])[0]]
                #weight = args.edge_weight.T[np.where(args.edge_index[0] == ucv_indices[0][0])[0]]
                #if j != 0:
                #    unit_cartesian_vector_idx = np.vstack((unit_cartesian_vector_idx, ucv_indices))
                #    edge_index_filtered = np.hstack((edge_index_filtered,edge_index_indices.T))
                #    edge_weight_filtered = np.hstack((edge_weight_filtered, weight.T))
                
                #else:
                #    unit_cartesian_vector_idx = ucv_indices
                #    edge_index_filtered = edge_index_indices.T
                #    edge_weight_filtered = weight

            #feature_array = torch.zeros((1, args.unit_cartesian_vector.T.shape[0], args.unit_cartesian_vector.T.shape[1]), dtype=torch.float32)
            #feature_array[0][unit_cartesian_vector_idx] = fmap2[:][unit_cartesian_vector_idx]
            #feature_array[0][edge_index_filtered[1]] = fmap2[:][edge_index_filtered[1]]
            
            #epipolar_line = epipolar_line[np.where(np.dot(epipolar_line, x_prime_vector_scaled.T[i])>0)]
            # for j in range(len(epipolar_line)):
            #     arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[j], 
            #                         imgWidth, imgHeight))
            #     cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)
            # cv2.imshow("image2",img2)
        
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
        #Build a dot product of normal_vector and all the points in the second image
        # coplanar_points_idx = np.where(abs(np.dot(unit_cartesian_vector.T, normal_vector))<0.01)
        # coplanar_points = (unit_cartesian_vector.T)[coplanar_points_idx]
        # epipolar_line_idx = np.where(np.sum(\
        #         np.cross(coplanar_points, x_prime_vector)*np.cross(coplanar_points, x_prime_vector_scaled), axis=1)<0)
        # epipolar_line = coplanar_points[epipolar_line_idx]
        # epipolar_line_idx = np.where(np.sum(\
        #         np.cross(x_prime_vector, x_prime_vector_scaled)*np.cross(epipolar_line, x_prime_vector_scaled), axis=1)>0)
        # epipolar_line = epipolar_line[np.where(np.dot(epipolar_line, x_prime_vector_scaled)>0)]
        return corr


    def plot_epipolar_line(epipolar_line):
        for i in range(len(epipolar_line)):
                #plot(cross_product1, cross_product2, coplanar_points, x_prime_vector, x_prime_vector_scaled, i)
                arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[i],
                                    imgWidth, imgHeight))
                cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)

    ##################################################################################################################
    def load_extrinsics(self, args):
        
        #Loading the extrinsics of Images
        R1,t1 = loadExtrinsicFile(args.extrinsics_path + args.filename1 + ".extr")
        R2,t2 = loadExtrinsicFile(args.extrinsics_path + args.filename2 + ".extr")

        args.R12 = np.dot(R2,R1.T)

        #t vector is always pointing outside from image 0 
        # so, to calculate the t vector from Image 2 to Image 0,
        args.t12 = np.dot(R2,(t1-t2))

        args.R21 = np.dot(R1,R2.T)
        #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 1 to Image 0
        #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
        args.t21 = np.dot(R1,(t2-t1))



##################################################################################################################
class EpipolarGeometryFeatureExtraction(torch.nn.Module):
    def __init__(self,params):
        super(EpipolarGeometryFeatureExtraction, self).__init__()
        self.conv1 = nn.ChebConv(params.epipolar_conv1_input_channels, params.epipolar_conv1_output_channels, params.conv1_kernel_size)
        self.relu1 = torch.nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu1(x)
        return x
