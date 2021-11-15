import numpy as np
import cv2
import math
import argparse
from coordinateconversion import CoordinateConversions as cc

def arguments_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_path",default="images/" ,help="Path of the Image folder")
    parser.add_argument("--extrinsics_path",default="extrinsics/",help="Path of the extrinsics folder")
    parser.add_argument("--divisions",default=5000,type=int,
           help="Divide 360 degrees by number of divisions to get angle of rotation around normal axis")
    parser.add_argument("--reference_image_number",default="0",
           help="The name of the reference image (all Rotations and translations are with respect to the reference image")
    parser.add_argument("--image1number",default="1",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--image2number",default="3",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--resize",type=bool,default=True,help="resizes the image for faster computation")
    args = parser.parse_args()

    return args

def load_image(args):
    global Image_Folderpath,filename0, filename1,filename2, extrinsic_Folderpath
    extrinsic_Folderpath=args.extrinsics_path
    Image_Folderpath=args.image_path
    image0name=str(args.reference_image_number)
    image1name=str(args.image1number)
    image2name=str(args.image2number)
    filename0=image0name.zfill(8)
    filename1=image1name.zfill(8)
    filename2=image2name.zfill(8)
    img1=cv2.imread(Image_Folderpath+filename1+".jpg")
    img2=cv2.imread(Image_Folderpath+filename2+".jpg")
    if args.resize:
        img1=cv2.resize(img1,(1080,540))
        img2=cv2.resize(img2,(1080,540))
    return img1, img2
    

def loadExtrinsicFile(ext_filename):
  # load the extrinsics text file into an np matrix
  lines = open(ext_filename, "r").readlines()
  data=[]
  for i in range(len(lines)):
      data.append(lines[i].rstrip('\n').split(','))
      
  if ((data[0][0]=="Dense3DKit_ExtrinsicData")& (data[1][0]=="R") & (data[5][0]=="t")):
      R=np.loadtxt(ext_filename,dtype=np.float64,skiprows=2,max_rows=3)
      t=np.loadtxt(ext_filename,dtype=np.float64,skiprows=6)
  
  return R,t

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def click(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Loading the R and t for image 1
        R1,t1=loadExtrinsicFile(extrinsic_Folderpath+filename1+".extr")
        #Loading the R and t for image 2
        R2,t2=loadExtrinsicFile(extrinsic_Folderpath+filename2+".extr")
        R=np.dot(R1.T,R2)
        #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 2 to Image 0, 
        t0=np.dot(-R2,t2)
        #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
        t=t0+t1
        t/=np.sqrt(np.dot(t, t))  #Normalizing the t vector
        #Converting the pixel co-ordinates into cartesian coordinates
        x_vector=np.asanyarray(cc.sphereMapCoordsToUnitCartesian(cc, x, y, imgWidth, imgHeight))  
        x_prime_vector=np.matmul(R,x_vector)
        normal_vector=np.cross(t, x_prime_vector) / np.linalg.norm(np.cross(t, x_prime_vector))  
        rotation_matrix_normalvec=rotation_matrix(normal_vector,2*math.pi/divisions)
        rotated_transvec=t
        for i in range(divisions):  #divisions
            rotated_transvec=np.dot(rotation_matrix_normalvec,rotated_transvec)
            #converting back to sphere map coordinates
            arc_Pixel=np.round(cc.unitCartesianToSphereMapCoords(cc, rotated_transvec, imgWidth,imgHeight))
            cv2.circle(img2, (int(arc_Pixel[0]),int(arc_Pixel[1])), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.imshow("image2",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def click2(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Loading the R and t for image 0
        R2,t2=loadExtrinsicFile(extrinsic_Folderpath+filename2+".extr")
        #Loading the R and t for the image 2
        R1,t1=loadExtrinsicFile(extrinsic_Folderpath+filename1+".extr")
        R=np.dot(R2.T,R1)
        #t vector is always pointing outside from image 0 so, to calculate the t vector from Image 1 to Image 0
        t0=np.dot(-R1,t1)
        #From Image0, we have to translate to Image 1. For this, we add t1 vector to t vector to get final t vector
        t=t0+t2
        t/=np.sqrt(np.dot(t, t))        #Normalizing the t vector
        #Converting the pixel co-ordinates into cartesian coordinates
        x_vector=np.asanyarray(cc.sphereMapCoordsToUnitCartesian(cc, x, y, imgWidth, imgHeight)) 
        x_prime_vector=np.matmul(R,x_vector)
        normal_vector=np.cross(t, x_prime_vector) / np.linalg.norm(np.cross(t, x_prime_vector))  
        rotation_matrix_normalvec=rotation_matrix(normal_vector,2*math.pi/divisions)
        rotated_transvec=t
        for i in range(divisions):  #divisions
            rotated_transvec=np.dot(rotation_matrix_normalvec,rotated_transvec)
            #converting back to sphere map coordinates
            arc_Pixel=np.round(cc.unitCartesianToSphereMapCoords(cc, rotated_transvec, imgWidth,imgHeight))
            cv2.circle(img1, (int(arc_Pixel[0]),int(arc_Pixel[1])), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.imshow("image1",img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    global divisions, img1, img2, imgWidth, imgHeight
    args=arguments_parser()
    divisions=args.divisions
    img1,img2=load_image(args)
    #Getting the shape of the image
    imgHeight,imgWidth,channels=img1.shape
    #Displaying the image
    cv2.namedWindow("image1")
    cv2.namedWindow("image2")
    cv2.setMouseCallback("image1", click)
    cv2.setMouseCallback("image2", click2)
    # keep looping until the 'esc' key is pressed
    # display the image and wait for a keypress
    cv2.imshow("image1", img1)
    cv2.imshow("image2",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()