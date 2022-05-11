import torch
import numpy as np
import cv2
import argparse
import healpy as hp
import time

from zmq import device

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
    parser.add_argument("--image2number", default="3",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--resize", type=bool, default=True, help="resizes the image for faster computation")
    parser.add_argument("--NSIDE", type=int, default=256, help="Number of sides in healpix pixellation" )
    args = parser.parse_args()

    return args

#####################################################################################################################
#Calculate epipolar line

def epipolar_line_cal(R, t):
    t_normalized = (t / np.sqrt(np.dot(t, t))).reshape(1,3)  #Normalizing the t vector
    t_normalized = torch.nn.functional.normalize(t, dim=0).reshape(1,3)  #Normalizing the t vector

    #Converting the pixel co-ordinates into cartesian coordinates
    x_vector = unit_cartesian_vector1 

    #Scaling the x_vector to minimum and maximum depth
    x_vector_scaled = x_vector * max_depth
    x_vector = x_vector * min_depth

    #Transforming the x_vector from Image 1 to Image 2
    #x_prime_vector = np.matmul(R, x_vector) + t[:, None]
    #x_prime_vector = x_prime_vector / np.sqrt(np.sum(x_prime_vector * x_prime_vector, axis = 0))
    #x_prime_vector_scaled = np.matmul(R, x_vector_scaled) + t[:, None]
    #x_prime_vector_scaled /= np.sqrt(np.sum(x_prime_vector_scaled * x_prime_vector_scaled, axis = 0))

    #Calculating the normal vector to define the plane
    #x_prime_vector = x_prime_vector.T

    # normal_vector = (np.cross(t_normalized[:, None, :], x_prime_vector[None, :, :])).reshape(-1,3)
    # normal_vector = normal_vector / np.sqrt(np.sum(normal_vector * normal_vector, axis = 1)).reshape(-1,1) 


    #Transforming the x_vector from Image 1 to Image 2
    x_prime_vector = torch.nn.functional.normalize(torch.matmul(R, x_vector) + t[:, None],dim =0)
    x_prime_vector_scaled = torch.nn.functional.normalize(torch.matmul(R, x_vector_scaled) + t[:, None], dim=0)

    #Calculating the normal vector to define the plane
    #x_prime_vector = x_prime_vector.T
    
    normal_vector = torch.nn.functional.normalize(torch.from_numpy(np.cross(t_normalized[None, :, :], x_prime_vector.T[None, :, :])), dim =0)
    te = time.time()
    print(te-ts)
        
    #Build a dot product of normal_vector and all the points in the second image
    # for i in range(25000,105100, 100):  #len(normal_vector)
    #     img1, img2 = load_image(args)
    #     coplanar_points_idx = np.where(abs(np.dot(unit_cartesian_vector1.T, normal_vector[i]))<0.01)
    #     coplanar_points = (unit_cartesian_vector1.T)[coplanar_points_idx]
    #     pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(unit_cartesian_vector1.T[i], 
    #                             imgWidth, imgHeight))
    #     cv2.circle(img1, (int(pixel[0]), int(pixel[1])), radius=10, color=(255, 0, 0), thickness=-1)
    #     cv2.imshow("image1",img1)
    #     epipolar_line_idx = np.where(np.sum(\
    #         np.cross(coplanar_points, x_prime_vector[i].T)*np.cross(coplanar_points, x_prime_vector_scaled.T[i]), axis=1)<0)
    #     epipolar_line = coplanar_points[epipolar_line_idx]
    #     epipolar_line = epipolar_line[np.where(np.dot(epipolar_line, x_prime_vector_scaled.T[i])>0)]
    #     for j in range(len(epipolar_line)):
    #         arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[j], 
    #                             imgWidth, imgHeight))
    #         cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)
    #     cv2.imshow("image2",img2)

    

    for i in range(25000,105100, 1000):  #len(normal_vector)
        img1, img2 = load_image(args)
        cv2.imshow("image1",img1)
        cv2.imshow("image2",img2)
        val = int(input("Enter your value: "))
        coplanar_points_idx = torch.where(abs(torch.matmul(unit_cartesian_vector1.T, normal_vector[0][val]))<0.01)
        coplanar_points = (unit_cartesian_vector1.T)[coplanar_points_idx]
        arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(x_prime_vector.T[val], 
                                imgWidth, imgHeight))
        cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=10, color=(0, 0, 255), thickness=-1)
        arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(x_prime_vector_scaled.T[val], 
                                imgWidth, imgHeight))
        cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=10, color=(0, 255, 0), thickness=-1)
        pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(unit_cartesian_vector1.T[val], 
                                imgWidth, imgHeight))
        cv2.circle(img1, (int(pixel[0]), int(pixel[1])), radius=10, color=(255, 0, 0), thickness=-1)
        cv2.imshow("image1",img1)
        #torch.broadcast_to()
        epipolar_line_idx = torch.where(torch.sum(\
            torch.cross(coplanar_points, torch.broadcast_to(x_prime_vector.T[val], (coplanar_points.shape[0],3)))*\
                torch.cross(coplanar_points, torch.broadcast_to(x_prime_vector_scaled.T[val], (coplanar_points.shape[0],3))), dim=-1)<0)
        epipolar_line = coplanar_points[epipolar_line_idx]
        epipolar_line = epipolar_line[torch.where(torch.matmul(epipolar_line, x_prime_vector_scaled.T[val])>0)]
        for j in range(len(epipolar_line)):
            arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[j], 
                                imgWidth, imgHeight))
            cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)
        cv2.imshow("image2",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #epipolar_line_idx = np.where(np.sum(\
     #       np.cross(coplanar_points, x_prime_vector)*np.cross(x_prime_vector_scaled, coplanar_points), axis=1)<0)
    #epipolar_line = coplanar_points[epipolar_line_idx]
    #epipolar_line = epipolar_line[np.where(np.dot(epipolar_line, x_prime_vector_scaled)>0)]
    return epipolar_line


####################################################################################################################
#Handle clicking events from opencv


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        

        
        
        #if arc_pixel.shape[1] != 2:
        #   arc_pixel = arc_pixel.reshape(-1, 2) 
        for i in range(len(epipolar_line)):
            #cross_product1 = np.cross(coplanar_points[i], x_prime_vector_scaled)
            #cross_product2 = np.cross(x_prime_vector, coplanar_points[i])
            #dot_product = np.dot(cross_product1, cross_product2)
            #plot(cross_product1, cross_product2, x_prime_vector, x_prime_vector_scaled, coplanar_points, i)
            arc_pixel = np.round(coord_convert.unitCartesianToSphereMapCoords(epipolar_line[i], 
                                imgWidth, imgHeight))
            cv2.circle(img2, (int(arc_pixel[0]), int(arc_pixel[1])), radius=1, color=(255, 0, 0), thickness=-1)

        cv2.imshow("image2",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def click2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        R = np.dot(R1,R2.T)
        #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 1 to Image 0
        #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
        t = np.dot(R1,(t1-t2))
        
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
#tensor reshaping and image vectorization




#################################################################################################################
#Load Image, extrinsics, reshape image, calculate phi, theta for given Healpix resolution


def main():
    global img1, img2, imgWidth, imgHeight,\
         coord_convert, unit_cartesian_vector1, theta, max_depth, min_depth, R1, t1, R2, t2, ts, te, device, args

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = time.time()
    args = arguments_parser()
    max_depth = args.max_depth
    min_depth = args.min_depth
    NSIDE = args.NSIDE

    #loading the image
    img1, img2 = load_image(args)
    #img1 = torch.tensor(img1, device = device).contiguous()
    
    #img2 = torch.tensor(img2, device = device).contiguous()

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
    unit_cartesian_vector1 = np.asanyarray(coord_convert.sphericalToCartesian(phi, theta, 1))
    unit_cartesian_vector1 = torch.tensor(coord_convert.sphericalToCartesian(phi, theta, 1))

    #R12 = np.matmul(R2,R1.T)
    R12 = torch.matmul(R2,R1.T)

    #t vector is always pointing outside from image 0 
    # so, to calculate the t vector from Image 2 to Image 0,
    #t12 = np.matmul(R2,(t1-t2))
    t12 = torch.matmul(R2,(t1-t2))

    #From Image0, we have to translate to Image 1. 
    #For this, we add t1 vector to t vector to get final t vector

    #R21 = np.matmul(R1,R2.T)
    R21 = torch.matmul(R1,R2.T)
    #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 1 to Image 0
    #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
    #t21 = np.dot(R1,(t2-t1))
    t21 = torch.matmul(R1,(t2-t1))

    # create red image
    #img = np.full((img1.shape[0],img1.shape[1],img1.shape[2]), (0,0,255), dtype=np.uint8)

    # convert to grayscale
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    # get coordinates (x,y)
    #xy_coords = np.flip(np.column_stack(np.where(gray >= 0)), axis=1)
    
    epipolar_line_img1 = epipolar_line_cal(R12, t12)
    epipolar_line_img2 = epipolar_line_cal(R21, t21, xy_coords[:,1], xy_coords[:,0])
    

##############################################################################################################

if __name__=="__main__":
    main()