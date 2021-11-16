import numpy as np
from constants import Constants
class CoordinateConversions:

    def __init__(self) -> None:
        self.constant=Constants()

    def sphereMapCoordsToSpherical(self,x,y,imgWidth,imgHeight):
        # Compute spherical theta coordinate
        theta = (1. - (x + .5) / imgWidth) * self.constant.pi2
        # Now theta is in [0, 2Pi]
        # Compute spherical phi coordinate
        phi = ((y + .5) * self.constant.pi) / imgHeight
        #Now phi is in [0, Pi]
        return (phi, theta)

    def sphericalToCartesian(self, phi, theta, radius):
        return radius * (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))

    def sphereMapCoordsToUnitCartesian(self, x, y, imgWidth, imgHeight):
        (phi, theta) = self.sphereMapCoordsToSpherical(x, y, imgWidth, imgHeight)
        return self.sphericalToCartesian(phi, theta, 1)

    def sphericalToSphereMapCoords(self, phi,theta, imgWidth, imgHeight):
        x=imgWidth * (1. - (theta * self.constant.KPi2Inverted)) - .5
        y=(imgHeight * phi) * self.constant.KPiInverted - .5
        return x,y

    def unitCartesianToSphereMapCoords(self, vec, imgWidth,imgHeight):
        if vec[2]<-1:
            vec[2]=-1
        if vec[2]>1:
            vec[2]=1
        if abs(vec[2])==1:
            KTheta=0
        else:
            KTheta=np.arctan2(vec[1], vec[0]) * self.constant.KPi2Inverted
        if (KTheta < 0):
            x = -KTheta * imgWidth - .5;           # -0.5 < x <= imgWidth/2 - 0.5 (left half of the pixel map)
        else:
            x = (1. - KTheta) * imgWidth - .5;     #imgWidth/2 - 0.5 <= x <= imgWidth - 0.5 (right half of the pixel map)
        KPhi=min(np.arccos(vec[2]) * self.constant.KPiInverted,1)
        y = KPhi * imgHeight - .5
        return x,y


