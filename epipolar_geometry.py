import numpy as np
import cv2
import glob
import math
import argparse


from numpy.core.numeric import NaN

from coordinateconversion import CoordinateConversions as cc

def arguments_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_path",default="images/" ,help="Path of the Image folder")
    parser.add_argument("--extrinsics_path",default="extrinsics/",help="Path of the extrinsics folder")
    parser.add_argument("--divisions",default=5000,type=int,
           help="Divide 360 degrees by number of divisions to get angle of rotation around normal axis")
    parser.add_argument("--image1number",default="0",
        help="The name of first image (give only the number eg 1 or 2)")
    parser.add_argument("--image2number",default="1",
        help="The name of first image (give only the number eg 1 or 2)")
    args = parser.parse_args()

    return args

def load_image(args):
    global Image_Folderpath,filename1,filename2, extrinsic_Folderpath
    extrinsic_Folderpath=args.extrinsics_path
    Image_Folderpath=args.image_path
    image1name=str(args.image1number)
    image2name=str(args.image2number)
    filename1=image1name.zfill(8)
    filename2=image2name.zfill(8)
    img1=cv2.imread(Image_Folderpath+filename1+".jpg")
    img2=cv2.imread(Image_Folderpath+filename2+".jpg")
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

  #t vector is from image 0 to image 1
  t=np.dot(-R,t)
  t/=np.sqrt(np.dot(t, t))
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


def computeRotationalMatrixForNormalVector(normalized_normalvec,trans_vec,divisions):
    #angle=np.deg2rad(360/divisions)
    angle=np.deg2rad(10)
    #normalized_normalvec=normal_vector/np.sqrt(np.sqrt(np.sum(normal_vector**2)))
    normalized_normalvec_skew=np.array([[0, -normalized_normalvec[2], normalized_normalvec[1]], [normalized_normalvec[2], 0, -normalized_normalvec[0]], 
                                        [-normalized_normalvec[1], normalized_normalvec[0], 0]])   
    
    #rotation_matrix_normalvec=np.array[]B9HFPz8nB9HFPz8n
    
    rotation_matrix_normalvec=np.eye(3)+np.sin(angle)*normalized_normalvec_skew+(1-np.cos(angle))*np.matmul(normalized_normalvec_skew,normalized_normalvec_skew)
    return rotation_matrix_normalvec
    

def click(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Loading the R and t for the image 2
        R,t=loadExtrinsicFile(extrinsic_Folderpath+filename2+".extr")
        
        #tvec_mark=np.round(cc.unitCartesianToSphereMapCoords(cc, t, imgWidth,imgHeight)) 

        #Converting the pixel co-ordinates into cartesian coordinates
        x_vector=np.asanyarray(cc.sphereMapCoordsToUnitCartesian(cc, x, y, imgWidth, imgHeight))  
        
        x_prime_vector=np.matmul(R,x_vector)
        #x_primevector_mark=np.round(cc.unitCartesianToSphereMapCoords(cc, x_prime_vector, imgWidth,imgHeight))
        
        normal_vector=np.cross(t, x_prime_vector) / np.linalg.norm(np.cross(t, x_prime_vector))  
        

        rotation_matrix_normalvec=rotation_matrix(normal_vector,2*math.pi/divisions)
        
        rotated_transvec=t
        for i in range(divisions):  #divisions
            rotated_transvec=np.dot(rotation_matrix_normalvec,rotated_transvec)
            #converting back to sphere map coordinates
            arc_Pixel=np.round(cc.unitCartesianToSphereMapCoords(cc, rotated_transvec, imgWidth,imgHeight))
            
            cv2.circle(img2, (int(arc_Pixel[0]),int(arc_Pixel[1])), radius=1, color=(0, 255, 0), thickness=-1)
        #    arc_Pixel_Stack[i]=arc_Pixel
        #cv2.circle(img2, (int(x_primevector_mark[0]),int(x_primevector_mark[1])), radius=10, color=(255, 0, 0), thickness=-1)
            
        #cv2.circle(img2, (int(tvec_mark[0]),int(tvec_mark[1])), radius=10, color=(0, 0, 255), thickness=-1)
        #img2=cv2.polylines(img2, np.int32([arc_Pixel_Stack]), isClosed = False,color = (0,255,0),thickness = 3)

        #cv2.drawContours(img2, arc_Pixel, 0, (255,255,255), 2)
        cv2.imshow("image2",img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def click2(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Loading the R and t for the image 2
        R,t=loadExtrinsicFile(extrinsic_Folderpath+filename1+".extr")

        #constructing a skew matrix
        translation_skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])              
        x_vector=np.asanyarray(cc.sphereMapCoordsToUnitCartesian( x, y, imgWidth, imgHeight))  #Converting the pixel co-ordinates into cartesian coordinates
        normal_vector=np.dot(translation_skew,x_vector)  

        #Rotating the translation vector around the normal vector
        rotational_matrix_normalvec=computeRotationalMatrixForNormalVector(normal_vector,divisions=1000)
        rotated_transvec=np.dot(rotational_matrix_normalvec,t)

        #converting back to sphere map coordinates
        arc_Pixel=cc.unitCartesianToSphereMapCoords(rotated_transvec, imgWidth,imgHeight)
        cv2.polylines(img2, (arc_Pixel[0],arc_Pixel[1]), isClosed = False,color = (0,255,0),thickness = 3, linetype = cv2.LINE_AA)
        cv2.drawContours(img1, arc_Pixel, 0, (255,255,255), 2)
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
    arc_Pixel_Stack=np.empty((divisions,2))
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