import cv2
import torch
import numpy as np

def load_image(args):
    #global Image_Folderpath,filename0, filename1, filename2, extrinsic_Folderpath
    if args.image_path == "images/":
        img1 = cv2.imread(args.image_path+args.filename1+".jpg")
        img2 = cv2.imread(args.image_path+args.filename2+".jpg")
        if args.resize:
            img1 = cv2.resize(img1, (1080,540))
            img2 = cv2.resize(img2, (1080,540))
    elif args.image_path == "processed_images/":
        img1 = np.load(args.image_path+args.filename1+".npy")
        img2 = np.load(args.image_path+args.filename2+".npy")

    
    return img1, img2

def loadExtrinsicFile(ext_filename):
  # load the extrinsics text file into an np matrix
  lines = open(ext_filename, "r").readlines()
  data = []
  for i in range(len(lines)):
      data.append(lines[i].rstrip('\n').split(','))
      
  if ((data[0][0]=="Dense3DKit_ExtrinsicData")& (data[1][0]=="R") & (data[5][0]=="t")):
      R = np.loadtxt(ext_filename, dtype=np.float64, skiprows=2, max_rows=3)
      t = np.loadtxt(ext_filename, dtype=np.float64, skiprows=6)
  
  return R,t

