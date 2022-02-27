import json
import healpy as hp
import numpy as np
import argparse
import os

from pygsp.graphs.nngraphs.sphereicosahedral import SphereIcosahedral
from healpix_sampling import calNpix

def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="groundtruth_json_files/", help="Path of the groundtruth json folder")
    parser.add_argument("--filename", default="sample.json", help="Name of the json file")
    parser.add_argument("--type", default="dodecohedron", help="Type of healpix pixellation")
    parser.add_argument("--k", type=int, default=5, help="Number of sides in healpix pixellation" )
    args = parser.parse_args()

    return args


def dodecohedron(args):
    
    ipix, NPIX=calNpix(2**args.k)
    phi, theta = np.degrees(hp.pix2ang(nside=2**args.k, ipix=ipix))
    phi = np.array(phi)
    theta = np.array(theta)
    keypts = np.empty((phi.size + theta.size,), dtype=phi.dtype)
    keypts[0::2] = phi
    keypts[1::2] = theta
    keypts = keypts.tolist()
    return keypts

def icosahedron(args):

    icosahedral_graph = SphereIcosahedral(2**args.k)
    phi = icosahedral_graph.signals["lat"]
    theta = icosahedral_graph.signals["lon"]
    keypts = np.empty((phi.size + theta.size,), dtype=phi.dtype)
    keypts[0::2] = phi
    keypts[1::2] = theta
    keypts = keypts.tolist()
    return keypts

    

def create_json(args):
    if args.type == "dodecohedron":
        keypts = dodecohedron(args)

    elif args.type == "icosahedron":
        keypts = icosahedron(args)

    else:
        print("please select either dodecohedron or icosahedron")

    
    interface_dict = {'keypointCoords': keypts}
    groundtruthfile = json.dumps(interface_dict)
    if not os.path.exists(args.file_path):
        os.mkdir(args.file_path)

    with open(args.file_path+args.filename, "w") as outfile:
        outfile.write(groundtruthfile)

def main():
    args = arguments_parser()
    create_json(args)

if __name__=="__main__":
    main()