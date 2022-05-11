import numpy as np
import healpy as hp
import cv2
import os
import math
import argparse
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix

from utils.coordinateconversion import CoordinateConversions


def arguments_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_path",default="images/" ,help="Path of the Image folder")
    parser.add_argument("--output_path",default="processed_images/32/",help="Path of the processed numpy image folder")
    parser.add_argument("--NSIDE",default=512,type=int,help="Number of subdivisions of dodecohedron")
    parser.add_argument("--resize", type=bool, default=False, help="resizes the image for faster computation")
    parser.add_argument("--imgWidth",default=14142,type=int, help="Width of the input image")
    parser.add_argument("--imgHeight",default=7071,type=int, help="Height of the input image")
    args = parser.parse_args()
    return args

def calNpix(NSIDE):
    print(
        "Approximate resolution at NSIDE {} is {:.2} deg".format(
            NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60))
    NPIX = hp.nside2npix(NSIDE)
    print(NPIX)
    ipix = np.arange(NPIX)
    return ipix, NPIX

def samplingImage(args, image_path,output_path,x,y):
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img=cv2.imread(image_path+filename)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.resize:
                img = cv2.resize(img, (args.imgWidth, args.imgHeight))
            array=img[y,x]
            numpy_filename=filename.split(".")[0]
            np.save(output_path+numpy_filename,array)
            

def main():
    coord_convert=CoordinateConversions()
    args=arguments_parser()
    NSIDE=args.NSIDE
    imgWidth=args.imgWidth
    imgHeight=args.imgHeight
    image_path=args.image_path
    output_path=args.output_path
    ipix, NPIX=calNpix(NSIDE)
    #Getting the values of phi and theta for all the pixels
    phi, theta = np.degrees(hp.pix2ang(nside=NSIDE, ipix=ipix)) #where phi is between 0 and 180, theta is 
                                                          #between 0 and 360 
                                                            
    #subdivisions = int(math.sqrt(NPIX / 12))
    #G = SphereHealpix(subdivisions, nest=True, k=20)
    x,y=coord_convert.sphericalToSphereMapCoords(np.deg2rad(phi),np.deg2rad(theta),imgWidth,imgHeight)
    x=x.astype(int)
    y=y.astype(int)
    samplingImage(args, image_path,output_path,x,y)

if __name__=="__main__":
    main()