import numpy as np
import cv2
import argparse
import time
import healpy as hp
import torch
import sys
sys.path.append("..")

from utils.coordinateconversion import CoordinateConversions
from utils.healpix_sampling import calNpix
from matlab_visualization import plot
from load_image import load_image, loadExtrinsicFile

##################################################################################################################
#Define arguments

def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="images/", help="Path of the Image folder")
    parser.add_argument("--extrinsics_path", default="extrinsics/", help="Path of the extrinsics folder")
    parser.add_argument("--max_depth", default=30, type=int, help = "Maximum depth of the image")
    parser.add_argument("--min_depth", default=1, type=int, help = "Minimum depth of the image")
    parser.add_argument("--divisions", default=15000, type=int,
           help="Divide 360 degrees by number of divisions to get angle of rotation around normal axis")
    parser.add_argument("--reference_image_number", default="0",
           help="The name of the reference image (all Rotations and translations are with respect to the reference image")
    parser.add_argument("--image1number", default="5",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--image2number", default="1",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--resize", type=bool, default=True, help="resizes the image for faster computation")
    parser.add_argument("--NSIDE", type=int, default=256, help="Number of sides in healpix pixellation" )
    args = parser.parse_args()

    return args

#####################################################################################################################
#Calculate epipolar line

def epipolar_line_cal(R, t, x_vector, y):
    t_normalized = torch.nn.functional.normalize(t, dim=0)  #Normalizing the t vector

    #Converting the pixel co-ordinates into cartesian coordinates
    #x_vector = torch.tensor(coord_convert.sphereMapCoordsToUnitCartesian(x, y, imgWidth, imgHeight)) 
    #x_vector = unit_cartesian_vector1

    #Scaling the x_vector to minimum and maximum depth
    x_vector_scaled = x_vector * max_depth
    x_vector = x_vector * min_depth

    #Transforming the x_vector from Image 1 to Image 2
    x_prime_vector = torch.nn.functional.normalize(torch.matmul(R, x_vector.T) + t[:, None],dim =0)
    x_prime_vector_scaled = torch.nn.functional.normalize(torch.matmul(R, x_vector_scaled.T) + t[:, None], dim=0)

    #Calculating the normal vector to define the plane
    #normal_vector = torch.nn.functional.normalize(torch.cross(t_normalized, x_prime_vector), dim=0) 
    normal_vector = torch.nn.functional.normalize(torch.from_numpy(np.cross(t_normalized, x_prime_vector.T[None, :, :])), dim =0) 
        
    #Build a dot product of normal_vector and all the points in the second image
    coplanar_points_idx = torch.where(abs(torch.sum(unit_cartesian_vector1 * normal_vector))<0.01)
    coplanar_points = (unit_cartesian_vector1)[coplanar_points_idx]
    epipolar_line_idx = torch.where(torch.sum(\
            torch.cross(coplanar_points, torch.broadcast_to(x_prime_vector, (coplanar_points.shape[0],3)))*torch.cross(coplanar_points, torch.broadcast_to(x_prime_vector_scaled, (coplanar_points.shape[0],3))), axis=1)<0)
    epipolar_line = coplanar_points[epipolar_line_idx]
    epipolar_line = epipolar_line[torch.where(torch.matmul(epipolar_line, x_prime_vector_scaled)>0)]
    return epipolar_line, coplanar_points, x_prime_vector, x_prime_vector_scaled


####################################################################################################################
#Handle clicking events from opencv


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img1, img2 = load_image(args)
        R = torch.matmul(R2,R1.T)

        #t vector is always pointing outside from image 0 
        # so, to calculate the t vector from Image 2 to Image 0,
        t = torch.matmul(R2,(t1-t2))

        #From Image0, we have to translate to Image 1. 
        #For this, we add t1 vector to t vector to get final t vector
        #val = int(input("Enter your value: "))
        epipolar_line, coplanar_points, x_prime_vector, x_prime_vector_scaled = epipolar_line_cal(R, t, unit_cartesian_vector1, y)
        arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(x_prime_vector, 
                                imgWidth, imgHeight))
        cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=10, color=(0, 0, 255), thickness=-1)
        arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(x_prime_vector_scaled, 
                                imgWidth, imgHeight))
        cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=10, color=(0, 255, 0), thickness=-1)
        arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(unit_cartesian_vector1, 
                                imgWidth, imgHeight))
        cv2.circle(img1, (int(arc_pixel[0]), int(arc_pixel[1])), radius=10, color=(0, 255, 0), thickness=-1)
        
        
        
        #if arc_pixel.shape[1] != 2:
        #   arc_pixel = arc_pixel.reshape(-1, 2) 
        for i in range(len(epipolar_line)):
            cross_product1 = np.cross(coplanar_points[i], x_prime_vector_scaled)
            cross_product2 = np.cross(coplanar_points[i], x_prime_vector)
            dot_product = np.dot(cross_product1, cross_product2)
            #plot(cross_product1, cross_product2, coplanar_points, x_prime_vector, x_prime_vector_scaled, i)
            arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[i], 
                                imgWidth, imgHeight))
            cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)
        cv2.imshow("image1",img1)
        cv2.imshow("image2",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def click2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        R = np.dot(R1,R2.T)
        #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 1 to Image 0
        #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
        t = np.dot(R1,(t2-t1))
        
        epipolar_line = epipolar_line_cal(R, t, x, y)
        
        for i in range(len(epipolar_line)):  #divisions
            #converting back to sphere map coordinates
            arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[i], 
                                imgWidth, imgHeight))
            cv2.circle(img1, (int(arc_pixel[0]), int(arc_pixel[1])),
                                radius=1, color=(0, 255, 0), thickness=-1)
        cv2.imshow("image1", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

##################################################################################################################
#Load Image, extrinsics, reshape image, calculate phi, theta for given Healpix resolution


def main():
    global img1, img2, imgWidth, imgHeight,\
         coord_convert, unit_cartesian_vector1, theta, max_depth, min_depth, R1, t1, R2, t2, args

    
    args = arguments_parser()
    max_depth = args.max_depth
    min_depth = args.min_depth
    NSIDE = args.NSIDE

    #loading the image
    img1, img2 = load_image(args)

    #Getting the shape of the image
    imgHeight, imgWidth, channels = img1.shape
    coord_convert = CoordinateConversions()

    #Loading the extrinsics of Images
    R1,t1 = loadExtrinsicFile(args.extrinsics_path + args.filename1 + ".extr")
    R2,t2 = loadExtrinsicFile(args.extrinsics_path + args.filename2 + ".extr")
    
    ipix, NPIX=calNpix(NSIDE)
    #Getting the values of phi and theta for all the pixels
    phi, theta = np.degrees(hp.pix2ang(nside=NSIDE, ipix=ipix)) #where phi is between 0 and 180, theta is 
                                                          #between 0 and 360 

    #Converting all the points in the image (healpix image) into unit cartesian coordinates
    unit_cartesian_vector1 = torch.from_numpy(np.asanyarray(coord_convert.sphericalToCartesian(phi, theta, 1)).T)
    
    #Displaying the image
    cv2.namedWindow("image1")
    cv2.namedWindow("image2")
    cv2.setMouseCallback("image1", click)
    cv2.setMouseCallback("image2", click2)
    # keep looping until the 'esc' key is pressed
    # display the image and wait for a keypress
    cv2.imshow("image1", img1)
    cv2.imshow("image2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##############################################################################################################

if __name__=="__main__":
    main()