import numpy as np
import math
class CoordinateConversions:

    def __init__(self) -> None:
        #self.pi=0.005555556
        #self.KPiInverted = 0.005555556
        #self.pi2=0.002777778
        #self.KPi2Inverted= 0.002777778
        pass

    def sphereMapCoordsToSpherical(self,x,y,imgWidth,imgHeight):
        # Compute spherical theta coordinate
        pi=math.pi
        pi2=2*pi
        
        theta = (1. - (x + .5) / imgWidth) * pi2
        # Now theta is in [0, 2Pi]
        # Compute spherical phi coordinate
        phi = ((y + .5) * pi) / imgHeight
        #Now phi is in [0, Pi]
        return (phi, theta)

    def sphericalToCartesian(self, phi, theta, radius):
        return radius * (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))

    def sphereMapCoordsToUnitCartesian(self, x, y, imgWidth, imgHeight):
        (phi, theta) = self.sphereMapCoordsToSpherical(self, x, y, imgWidth, imgHeight)
        return self.sphericalToCartesian(self, phi, theta, 1)

    def sphericalToSphereMapCoords(self, phi,theta, imgWidth, imgHeight):
        
        KPi2Inverted= 0.159235669
        KPiInverted=2*KPi2Inverted
        x=imgWidth * (1. - (theta * KPi2Inverted)) - .5
        y=(imgHeight * phi) * KPiInverted - .5
        #x=np.round(x)
        #y=np.round(y)
        #x=x.astype(int)
        #y=y.astype(int)
        return x,y

    def unitCartesianToSphereMapCoords(self, vec, imgWidth,imgHeight):
        KPi2Inverted= 0.159235669
        KPiInverted = KPi2Inverted*2
        if vec[2]<-1:
            vec[2]=-1
        if vec[2]>1:
            vec[2]=1
        if abs(vec[2])==1:
            KTheta=0
        else:
            KTheta=np.arctan2(vec[1], vec[0]) * KPi2Inverted
        if (KTheta < 0):
            x = -KTheta * imgWidth - .5;           # -0.5 < x <= imgWidth/2 - 0.5 (left half of the pixel map)
        else:
            x = (1. - KTheta) * imgWidth - .5;     #imgWidth/2 - 0.5 <= x <= imgWidth - 0.5 (right half of the pixel map)
        KPhi=min(np.arccos(vec[2]) * KPiInverted,1)
        y = KPhi * imgHeight - .5
        return x,y


