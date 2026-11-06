# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:13:27 2026

@author: 77891

Stress inversion and stress map toolkit.

Main building blocks:

* ``stressParameter`` - reduced stress tensor, fault geometry and the
  derived quantities (shear stress, normal stress, Mohr-Coulomb).
* ``meshgrid``        - 4-D inversion grid (Euler angles + stress ratio).
* ``geology`` / ``picture`` - map projection and plotting utilities.
* ``solution``        - fitness, attractor and peak-search inversion.
"""

import numpy as np
import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.transforms as transforms
import matplotlib.image as mpimg
from scipy import linalg
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D


def is_2d_array(arr):
    """Return True if *arr* is a 2-D numpy array or a list of lists."""
    if isinstance(arr, np.ndarray):
        return arr.ndim == 2
    elif isinstance(arr, list):
        return all(isinstance(row, list) for row in arr)
    else:
        return False

def CalAngle(v1, v2):
    """Angle that rotates *v1* onto *v2* (counter-clockwise positive).

    A clockwise rotation is returned as 2*pi - theta, so the value always
    lies in [0, 2*pi).
    """
    # Product of the moduli of the two vectors
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if TheNorm == 0:
        return 0
    # Cross product: sine of the angle, its sign gives the rotation sense
    rho = np.arcsin(np.cross(v1, v2) / TheNorm)
    # Dot product: cosine of the angle
    theta = np.arccos(np.dot(v1, v2) / TheNorm)
    if rho < 0:
        return 2 * np.pi - theta
    else:
        return theta
                
def add_right_cax(ax, pad, width):
    """Append a colorbar axes of the same height on the right of *ax*.

    pad   : spacing between *ax* and the new colorbar axes.
    width : width of the new colorbar axes.
    """
    axpos = ax.get_position()
    caxpos = transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

class StressLegendHandle:
    """Legend handle that only carries the dip angle to be rendered."""

    def __init__(self, angle):
        self.angle = angle


class HandlerStressLegend(HandlerBase):
    """Render one legend entry as a horizontal line plus a centre square.

    The line length encodes the dip angle, so the legend shows how the
    rotated squares of the figure map onto different dipping planes.
    """

    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent,
        width, height,
        fontsize, trans
    ):

        # ==========================================
        # Position of the square
        # ==========================================
        x0 = xdescent + width * 0.15
        y0 = ydescent + height * 0.50

        # ==========================================
        # Maximum line length
        # ==========================================
        Lmax = width * 0.75

        # ==========================================
        # Length depends on the dip angle
        # L = Lmax * cos(theta)
        # ==========================================
        theta = np.deg2rad(orig_handle.angle)

        L = Lmax * np.cos(theta)

        # ==========================================
        # The line always stays horizontal
        # ==========================================
        x1 = x0 + L
        y1 = y0

        # Horizontal line
        line = Line2D(
            [x0, x1],
            [y0, y1],
            color='black',
            linewidth=2,
            solid_capstyle='round',
            transform=trans
        )

        # Centre square
        square = Line2D(
            [x0],
            [y0],
            marker='s',
            markersize=8,
            markerfacecolor='black',
            markeredgecolor='black',
            linestyle='None',
            transform=trans
        )

        return [line, square]

class stressParameter:
    """Reduced stress tensor and fault geometry.

    All angles are stored in radians.  The class provides the quantities
    required by stress inversion: the 5-D stress vector, the normal and
    shear traction on a plane, the Mohr-Coulomb criterion and the slip
    vector of a fault.
    """

    def __init__(self,grid = np.array([0,0,0,0]),fault = np.deg2rad(np.array([227,80,172])),friction = 0.6):
        """Build a stress parameter object.

        alpha, beta, gamma are the Euler angles of the successive axis
        rotations (rotation order z, y, z), with 0 <= alpha <= 2*pi,
        0 <= beta <= pi/2 and 0 <= gamma <= pi.
        R is the stress shape ratio, R = (sigma1 - sigma2) / (sigma1 - sigma3),
        with 0 <= R <= 1.

        strike, dip, slip describe the fault: strike is the angle measured
        clockwise from north to the strike direction, dip is the angle from
        the horizontal plane down to the fault plane and slip is the slip
        angle between the strike direction and the slip direction of the
        hanging wall relative to the foot wall (measured counter-clockwise
        here).  0 <= strike <= 2*pi, 0 <= dip <= pi/2, -pi <= slip <= pi.
        """
        [self.__alpha, self.__beta, self.__gamma, self.__R] = grid
        [self.__strike, self.__dip, self.__slip] = fault
        self.__friction = friction
    
    def setGrid(self,newGrid):
        """Update the stress grid in radians (R is a ratio and stays as is)."""
        [self.__alpha, self.__beta, self.__gamma, self.__R] = newGrid
        
    def setGridInDeg(self,newGrid):
        """Update the stress grid from degree values (R stays a ratio)."""
        [self.__alpha, self.__beta, self.__gamma] = np.deg2rad(newGrid[:3])
        self.__R = newGrid[3]
        
    def setFaultInDeg(self,newFault):
        """Update the fault parameters from degree values."""
        [self.__strike, self.__dip, self.__slip] = np.deg2rad(newFault)
        
    def setFault(self,newFault):
        """Update the fault parameters from radian values."""
        [self.__strike, self.__dip, self.__slip] = newFault
        
    def setFriction(self,friction):
        """Update the friction coefficient used by :meth:`MC`."""
        self.__friction = friction
        
    def getStress(self):
        """Return the stress grid as [alpha, beta, gamma, R]."""
        return [self.__alpha, self.__beta, self.__gamma, self.__R]
    
    def getFault(self):
        """Return the fault parameters as [strike, dip, slip]."""
        return [self.__strike, self.__dip, self.__slip]
        
    def Rx(self,rad):
        """Rotation matrix about the x axis."""
        matrix = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
        return matrix
    
    def Ry(self,rad):
        """Rotation matrix about the y axis."""
        matrix = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
        return matrix
    
    def Rz(self,rad):
        """Rotation matrix about the z axis."""
        matrix = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        return matrix
    
    def Rzyz(self,rad1,rad2,rad3):
        """Composite z-y-z intrinsic rotation matrix."""
        return self.Rz(rad1) @ self.Ry(rad2) @ self.Rz(rad3)
    
    def Rccc(self,order,rad1,rad2,rad3):
        """Composite rotation matrix built from an arbitrary axis order.

        *order* holds three indices into [Rx, Ry, Rz] (0 = x, 1 = y, 2 = z)
        and rad1..rad3 are the matching rotation angles.
        """
        r = [self.Rx,self.Ry,self.Rz]
        [r1,r2,r3] = order
        return r[r1](rad1) @ r[r2](rad2) @ r[r3](rad3)
    
    def invariantValues(self,stressMatrix):
        """Return the three invariants (I1, I2, I3) of a stress matrix."""
        I1 = stressMatrix[0][0] + stressMatrix[1][1] + stressMatrix[2][2]
        I2 = stressMatrix[0][0] * stressMatrix[1][1] + stressMatrix[1][1] * stressMatrix[2][2] + stressMatrix[0][0] * stressMatrix[2][2] - stressMatrix[0][1]**2 - stressMatrix[0][2]**2 - stressMatrix[1][2]**2
        I3 = np.linalg.det(stressMatrix)
        return np.array([I1,I2,I3])
    
    def normalizationIn5D(self):
        """Normalise the stress tensor and rotate it back to the original axes.

        The tensor is first scaled to unit norm in the principal stress frame
        and afterwards rotated with the inverse z-y-z rotation.
        """
        normalizationMatrix = np.array([[2 - self.__R, 0, 0],[0, 2 * self.__R - 1, 0],[0, 0, -(self.__R + 1)]]) \
            / np.sqrt(3 * self.__R**2 - 3 * self.__R + 3)
        Rrotation = self.Rzyz(-self.__gamma, -self.__beta, -self.__alpha)
        normalizationMatrix = Rrotation.T @ normalizationMatrix @ Rrotation
        
        return normalizationMatrix
        
    def vector5D(self, stress = None):
        """Map a stress state onto the 5-D unit sphere of Sato & Yamaji (2006).

        With ``stress=None`` the current grid is converted; otherwise every
        row of *stress* is converted and an (n, 5) array is returned.
        """
        if stress is None:
            # Coefficients of the orthonormal 5-D basis
            a = -1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
            b = 1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
            c = 1 / np.sqrt(6)
            # Normalised stress tensor expressed in the original axes
            nM = self.normalizationIn5D()
            # First row plus the three independent remaining components
            [X11,X12,X13] = nM[0]
            X23 = nM[1,2]
            X22 = nM[1,1]
            X33 = nM[2,2]
            # Project the six independent components onto the 5-D basis
            vector = np.array([a * X11 + b * X22 + c * X33, b * X11 + a * X22 + c * X33, X23, X13, X12])
            return vector
        else:
            vector = np.array([])
            for i in stress:
                a = -1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
                b = 1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
                c = 1 / np.sqrt(6)
                # Take the next stress state from the input array
                self.setGrid(i)
                nM = self.normalizationIn5D()
                [X11,X12,X13] = nM[0]
                X23 = nM[1,2]
                X22 = nM[1,1]
                X33 = nM[2,2]
                vector = np.append(vector, [a * X11 + b * X22 + c * X33, b * X11 + a * X22 + c * X33, X23, X13, X12]) 
            # One row per input stress state
            vector = np.reshape(vector, (-1,5))
            return vector
    
    def distance5D(self, v1, v2):
        """Euclidean distance between two 5-D stress vectors."""
        dv = v1 - v2
        return np.linalg.norm(dv)
    
    def s5_to_stress_tensor(self, v5, normalize=False):
        """
        Convert a 5-D s-vector as defined by Sato & Yamaji (2006) into a
        3x3 reduced stress tensor.

        Parameters
        ----------
        v5 : array-like, shape (5,)
            Five-component stress vector.
        normalize : bool
            If True the input is first scaled to unit length.  This is
            especially useful for cluster centres such as K-means centroids.

        Returns
        -------
        S : ndarray, shape (3, 3)
            Symmetric reduced stress tensor.
        """
        v5 = np.asarray(v5, dtype=float)

        # Only a single 5-D vector is accepted here
        if v5.shape != (5,):
            raise ValueError("The input must be a vector of length 5")

        # Optional normalisation onto the unit sphere
        if normalize:
            norm = np.linalg.norm(v5)
            if norm == 0:
                raise ValueError("A zero vector cannot be normalised")
            v5 = v5 / norm

        # Basis coefficients, identical to the ones used by vector5D
        a = -1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
        b = 1 / (np.sqrt(2) * 2) - 1 / (np.sqrt(6) * 2)
        c = 1 / np.sqrt(6)
        
        [s1, s2, X23, X13, X12] = v5
        # Invert the two linear combinations to recover the tensor terms
        X22 = (s1 - (s1 - s2)*(a - c)/(a - b)) / (a + b - 2 * c)
        X11 = (s1 - s2)/(a - b) + X22
        X33 = (s1 - a * X11 - b * X22) / c
        # Assemble the symmetric reduced stress tensor
        S = np.array([[X11, X12, X13],
                      [X12, X22, X23],
                      [X13, X23, X33]])
        return S
    
    def s5Array_to_stress_tensor(self,v5,normalize = False):
        """Convert an (n, 5) array of s-vectors into (n, 3, 3) tensors."""
        v = np.array([])
        for i in v5:
            # Convert one row (one stress state) at a time
            i = self.s5_to_stress_tensor(i, normalize = normalize)
            v = np.append(v, i)
        v = np.reshape(v, (-1,3,3))
        return v    

    def strikeVector(self):
        """Strike unit vector expressed in the original coordinate frame."""
        strikeVector = np.dot(self.Rz(self.__strike),np.array([0,-1,0]))
        return strikeVector.flatten()
    
    def surfaceVector(self):
        """Unit vector of the fault plane, dipping by *dip* from the z axis."""
        surfaceVector = np.dot(self.Rz(self.__strike + np.pi / 2) @ self.Rx(-self.__dip), np.array([0,-1,0]))
        return surfaceVector.flatten()
    
    def faultNormalVector(self):
        """Unit normal of the fault plane (normal of the hanging wall)."""
        normalVector = np.cross(self.strikeVector(), self.surfaceVector()) \
            / np.linalg.norm(np.cross(self.strikeVector(), self.surfaceVector()))
        return normalVector.flatten()
    
    def directionCosine(self,vector,isDefaultRotation = True,order = (),rotation = ()):
        """Direction cosines of *vector* in the rotated coordinate frame.

        The coordinate frame is rotated while the vector keeps its position
        in space.  When *isDefaultRotation* is False the rotation described
        by *order* and *rotation* is used instead of the stored Euler angles.
        """
        if isDefaultRotation:
            # Inverse z-y-z rotation of the stored Euler angles
            newVector = np.dot(self.Rzyz(-self.__alpha, -self.__beta, -self.__gamma),vector)
            if np.abs(np.linalg.norm(newVector) - 1) < 0.01:
                return newVector.flatten()
            else:
                # Guard against small numerical drift
                cosine = newVector / np.linalg.norm(newVector)
                return cosine.flatten()
        else:
            (rad1,rad2,rad3) = rotation
            newVector = np.dot(self.Rccc(order, -rad1, -rad2, -rad3),vector)
            if np.abs(np.linalg.norm(newVector) - 1) < 0.01:
                return newVector.flatten()
            else:
                cosine = newVector / np.linalg.norm(newVector)
                return cosine.flatten()
        
    def shearStressVector(self,isDefaultRotation = True,order = (),rotation = (),R = 0.5,isUnit = True):
        """Shear traction on the hanging wall, in the principal axes frame.

        *R* and the rotation arguments override the stored stress ratio and
        coordinate frame.  When *isUnit* is True the traction is normalised,
        otherwise the raw (scaled) traction is returned.
        """
        if isDefaultRotation:
            [cosine1,cosine2,cosine3] = self.directionCosine(self.faultNormalVector())
            # Components of the shear traction in the principal frame
            shear = np.array([(1 - cosine1**2 - self.__R * cosine2**2) * cosine1,\
                             (self.__R - cosine1**2 - cosine2**2 * self.__R) * cosine2,\
                             (- cosine1**2 - cosine2**2 * self.__R) * cosine3])
            if isUnit:
                if np.linalg.norm(shear) == 0:
                    # Degenerate case: the shear traction vanishes
                    unitShear = np.NaN
                else:
                    unitShear = shear / np.linalg.norm(shear)
                return unitShear
            else:
                return shear
        else:
            [cosine1,cosine2,cosine3] = self.directionCosine(self.faultNormalVector(),isDefaultRotation = isDefaultRotation,order = order,rotation = rotation)
            shear = np.array([(1 - cosine1**2 - R * cosine2**2) * cosine1,\
                             (R - cosine1**2 - cosine2**2 * R) * cosine2,\
                             (- cosine1**2 - cosine2**2 * R) * cosine3])
            unitShear = shear / np.linalg.norm(shear)
            return unitShear
    
    def shearStressVectorTransform(self,isDefaultRotation = True,order = (),rotation = (),R = 0.5):
        """Rotate the shear traction back to the original coordinate frame."""
        if isDefaultRotation:
            # Forward z-y-z rotation
            return np.dot(self.Rzyz(self.__gamma, self.__beta, self.__alpha),self.shearStressVector())
        else:
            (rad1,rad2,rad3) = rotation
            # Forward custom rotation
            return np.dot(self.Rccc(order,rad1,rad2,rad3),self.shearStressVector(isDefaultRotation = isDefaultRotation,order = order,rotation = rotation,R = R))
    
    def shearStress(self):
        """Shear stress magnitude weighted by the Sato & Yamaji factor k2."""
        k2 = ((2 - self.__R) + (self.__R + 1)) / np.sqrt(2 * self.__R**2 - 2 * self.__R + 2)
        return self.shearStressVector(isUnit = False) * k2
    
    def normalStress(self):
        """Normal stress vector on the fault plane, weighted by k1 and k2."""
        k1 = (self.__R + 1) / np.sqrt(2 * self.__R**2 - 2 * self.__R + 2)
        k2 = ((2 - self.__R) + (self.__R + 1)) / np.sqrt(2 * self.__R**2 - 2 * self.__R + 2)
        [cosine1,cosine2,cosine3] = self.directionCosine(self.faultNormalVector())
        return np.array([cosine1, cosine2, cosine3]) * (k1 + k2 * (cosine1**2 + cosine2**2 * self.__R))      
    
    def MC(self):
        """Mohr-Coulomb failure function: |tau| - mu * |sigma_n|."""
        return np.linalg.norm(self.shearStress()) - self.__friction * np.linalg.norm(self.normalStress())
    
    def slipVector(self):
        """Slip vector of the hanging wall in the original coordinate frame."""
        Rrotation = self.Rzyz(self.__strike, -self.__dip, -self.__slip)
        slipVector = np.dot(Rrotation,np.array([0,-1,0]))
        return slipVector.flatten()
    
class meshgrid:
    """4-D inversion grid of Euler angles plus stress ratio.

    Directions are Fibonacci-distributed points (alpha, beta); every
    direction is combined with N2 values of gamma and N3 values of R.
    Only the hemisphere z >= 0 is sampled because a stress tensor and its
    negative describe the same physical state.
    """

    def __init__(self,N1=100,N2=100,N3=5,isRad = True):
        """Create a grid with N1 directions, N2 rotations and N3 ratios."""
        self.__N1 = N1
        self.__N2 = N2
        self.__N3 = N3
        self.__isRad = isRad
        
    def setN1(self,N1):
        """Set the number of sampling directions."""
        self.__N1 = N1
        
    def setN2(self,N2):
        """Set the number of rotations around every direction."""
        self.__N2 = N2
        
    def setN3(self,N3):
        """Set the number of stress ratios per rotation."""
        self.__N3 = N3
        
    def FibonacciGrid(self,isSimplify = False):
        """Fibonacci (spherical spiral) points on the upper hemisphere.

        The sequence uses the golden ratio so that the points are almost
        evenly spread; *isSimplify* keeps a coarser sampling.
        """
        goldenRatio = 0.618
        lib = np.array([])
        if isSimplify:
            # Coarser sampling
            factor = 4
        else:
            factor = 2
        for n in range(1,self.__N1 * factor + 1):
            zn = (2 * n - 1) / (self.__N1 * factor) - 1
            # Upper hemisphere only (the sign of the stress is irrelevant)
            if zn < 0:
                continue
            xn = np.sqrt(1 - zn**2) * np.cos(2 * np.pi * n * goldenRatio)
            if isSimplify and xn < 0:
                continue
            yn = np.sqrt(1 - zn**2) * np.sin(2 * np.pi * n * goldenRatio)
            lib = np.append(lib, [xn,yn,zn])
        lib = lib.reshape((-1,3))
        return lib
    
    def normalFibonacciGrid(self,isSimplify = False):
        """Fibonacci points on the lower hemisphere (z <= 0).

        Mirror image of :meth:`FibonacciGrid`; it is used when the
        down-going normal of a plane has to be sampled.
        """
        goldenRatio = 0.618
        lib = np.array([])
        if isSimplify:
            # Coarser sampling
            factor = 4
        else:
            factor = 2
        for n in range(1,self.__N1 * factor + 1):
            zn = (2 * n - 1) / (self.__N1 * factor) - 1
            # Lower hemisphere only
            if zn > 0:
                continue
            xn = np.sqrt(1 - zn**2) * np.cos(2 * np.pi * n * goldenRatio)
            if isSimplify and xn < 0:
                continue
            yn = np.sqrt(1 - zn**2) * np.sin(2 * np.pi * n * goldenRatio)
            lib = np.append(lib, [xn,yn,zn])
        lib = lib.reshape((-1,3))
        return lib
    
    def rotationGrid(self,FibonacciPoint):
        """Rotate around *FibonacciPoint* and sample N2 directions.

        A 180 degree fan is sampled in steps of 180 / N2 degrees.
        """
        if not isinstance(FibonacciPoint, np.ndarray):
            raise TypeError('The parameter FibonacciPoint must be a three-dimensional numpy.ndarray vector')
            
        [x0,y0,z0] = FibonacciPoint
        initialVector = np.array([[x0, y0 - 1 / y0, z0]])
        initialVector = initialVector / np.linalg.norm(initialVector)
        # The initial vector rotating around a Fibonacci grid point takes the unit direction vector intersecting the positive Y-axis
        
        # Rodrigues' rotation formula. The total rotation angle is 180 degree, counter-clockwise positive.
        gridAngle = np.deg2rad(180 / self.__N2)
        vector = np.copy(initialVector)
        for n in range(1,self.__N2 + 1):
            lib = np.cos(n * gridAngle) * initialVector + np.sin(n * gridAngle) * np.cross(FibonacciPoint,initialVector)
            vector = np.append(vector, [lib], axis = 0)
        return vector
    
    def CartesiusToEuler(self,FibonacciPoint):
        """Convert a Cartesian grid point into Euler angles (alpha, beta, gamma)."""
        [x0,y0,z0] = FibonacciPoint
        # Azimuth of the point measured around the z axis
        alpha = CalAngle([1,0],[x0,y0])
        # Polar angle measured from the z axis
        beta = np.arccos(np.dot(FibonacciPoint,[0,0,1]))
        # gamma = 0 for the first member of the fan
        EulerAngle = np.array([[alpha,beta,0]])
        gridAngle = np.deg2rad(180 / self.__N2)
        for n in range(1,self.__N2 + 1):
            # Fan of N2 values of gamma
            EulerAngle = np.append(EulerAngle, [np.array([alpha,beta,n * gridAngle])], axis = 0)
        return EulerAngle
    
    def RGrid(self,EulerAngle):
        """Attach the N3 stress ratio values in (0, 1] to one Euler angle triple."""
        if not isinstance(EulerAngle, np.ndarray):
            raise TypeError('The parameter EulerAngle must be a three-dimensional numpy.ndarray vector')
        
        # First entry: R = 0
        fourDimensionalVector = np.array([np.append(EulerAngle, 0)])
        for n in range(1,self.__N3 + 1):
            # N3 evenly spaced values of R
            lib = np.append(EulerAngle, 1 / self.__N3 * n)
            fourDimensionalVector = np.append(fourDimensionalVector, [lib], axis = 0)
        return fourDimensionalVector
    
    def fourDimensionalGrid(self,isSimplify = False):
        """Build the complete 4-D grid (alpha, beta, gamma, R)."""
        z = self.FibonacciGrid(isSimplify)
        Euler = np.array([[0,0,0]])
        for i in z:
            Euler = np.vstack((Euler,self.CartesiusToEuler(i)))
        # Drop the placeholder first row
        Euler = np.delete(Euler, 0, axis = 0)
        if self.__isRad:
            # Angles are kept in radians
            grid = np.array([[0,0,0,0]])
            for i in Euler:
                grid = np.vstack((grid,self.RGrid(i)))
            grid = np.delete(grid, 0, axis = 0)
        else:
            # Angles are converted to degrees
            Euler = np.rad2deg(Euler)
            grid = np.array([[0,0,0,0]])
            for i in Euler:
                grid = np.vstack((grid,self.RGrid(i)))
            grid = np.delete(grid, 0, axis = 0)
        return grid
        
class geology(stressParameter):
    """Plate boundary data and map projection helpers."""

    def __init__(self, platesBoundary = './plate/boundary.csv'):
        """Load the plate boundary points from a CSV file."""
        super().__init__()
        self.__boundary = pd.read_csv(platesBoundary,index_col=0)
        # Remove completely empty rows and columns
        self.__boundary = self.__boundary.dropna(axis=0, how="all")
        self.__boundary = self.__boundary.dropna(axis=1, how="all")
        self.__boundary = self.__boundary.values
        # Earth radius in km
        self.__rEarth = 6371
    
    def getBoundary(self):
        """Return the plate boundary points as an (n, 2) array."""
        return self.__boundary
    
    def getREarth(self):
        """Return the Earth radius used by the projection (km)."""
        return self.__rEarth
     
    def Mercator(self,LonAndLat):
        """Project (lon, lat) in degrees onto spherical Mercator (x, y) in km."""
        (lon,lat) = LonAndLat
        x = np.deg2rad(lon) * self.getREarth()
        y = np.log(np.tan(np.deg2rad(90 + lat) / 2)) * self.getREarth()
        return (x,y)
        
    def locTrans(self,LonAndLat,pictureX,pictureY,area):
        """Map (lon, lat) onto the pixel coordinates of a background image.

        *area* is (minLon, maxLon, minLat, maxLat), (pictureX, pictureY) is
        the image size in pixels and the y axis points downwards.
        """
        (minLon,maxLon,minLat,maxLat) = area
        # Mercator coordinates of the image corners
        (xmin,ymin) = self.Mercator([minLon,minLat])
        (xmax,ymax) = self.Mercator([maxLon,maxLat])
        (x,y) = self.Mercator(LonAndLat)
        # Normalise to the image size and flip the y axis
        X = (x - xmin) / (xmax - xmin) * pictureX
        Y = pictureY - (y - ymin) / (ymax - ymin) * pictureY
        return (X,Y)
    
    def matchAziWithDatabase(self, filePath, inputData,
                             area=(110, 131, 29, 46)):
        """
        Keep the better of two candidate azi / aziType pairs per fault.

        The nearest station of the stress database is used as reference: if
        it has a valid REGIME the candidate with the same regime is
        preferred, otherwise the candidate whose azimuth is closest to the
        database azimuth is selected.

        Parameters
        ----------
        filePath : str
            Path of the CSV database file.

        inputData : array-like, shape (n, 6)

            One row per fault:

            [
                azi1,
                aziType1,
                azi2,
                aziType2,
                faultLat,
                faultLon
            ]

            for example:

            [
                [120, 'Normal',     30, 'Strike_Slip', 35.2, 118.5],
                [80,  'Reverse',   170, 'Normal',      39.1, 121.2]
            ]

        area : tuple
            Database window:

            (minLon, maxLon, minLat, maxLat)

            default:

            (110, 131, 29, 46)

        Returns
        -------
        result : ndarray, shape (n, 3)

            One row per fault:

            [selectedAzi, selectedAziType, chosenCandidate]

            where chosenCandidate is 0 for the first and 1 for the second
            candidate.

        """

        import numpy as np
        import pandas as pd


        # =========================================================
        # 1. Read the database
        # =========================================================

        data = pd.read_csv(
            filePath,
            low_memory=False
        )


        # =========================================================
        # 2. Check the required columns
        # =========================================================

        requiredColumns = [
            'LAT',
            'LON',
            'AZI',
            'QUALITY',
            'REGIME'
        ]

        missingColumns = [
            column
            for column in requiredColumns
            if column not in data.columns
        ]

        if len(missingColumns) > 0:

            raise ValueError(
                "The database is missing the following columns: "
                + ", ".join(missingColumns)
            )


        # =========================================================
        # 3. Keep only the columns that are needed
        # =========================================================

        data = data[
            [
                'LAT',
                'LON',
                'AZI',
                'QUALITY',
                'REGIME'
            ]
        ].copy()


        # =========================================================
        # 4. Convert the values to numeric types
        # =========================================================

        data['LAT'] = pd.to_numeric(
            data['LAT'],
            errors='coerce'
        )

        data['LON'] = pd.to_numeric(
            data['LON'],
            errors='coerce'
        )

        data['AZI'] = pd.to_numeric(
            data['AZI'],
            errors='coerce'
        )


        # =========================================================
        # 5. Drop rows with an invalid latitude, longitude or AZI
        # =========================================================

        data = data.dropna(
            subset=[
                'LAT',
                'LON',
                'AZI'
            ]
        )


        # =========================================================
        # 6. Normalise the format of QUALITY and REGIME
        # =========================================================

        data['QUALITY'] = (
            data['QUALITY']
            .astype(str)
            .str.strip()
            .str.upper()
        )

        data['REGIME'] = (
            data['REGIME']
            .astype(str)
            .str.strip()
            .str.upper()
        )


        # =========================================================
        # 7. area
        # =========================================================

        minLon, maxLon, minLat, maxLat = area


        # =========================================================
        # 8. Filter:
        #
        # inside the area
        # +
        # QUALITY = A / B / C
        # =========================================================

        data = data[
            (data['LON'] >= minLon)
            &
            (data['LON'] <= maxLon)
            &
            (data['LAT'] >= minLat)
            &
            (data['LAT'] <= maxLat)
            &
            (data['QUALITY'].isin(['A', 'B', 'C']))
        ].copy()


        if len(data) == 0:

            raise ValueError(
                "No valid data with QUALITY A, B or C exists in the given area."
            )


        # =========================================================
        # 9. REGIME -> aziType
        #
        # NF -> Normal
        # TF -> Reverse
        # SS -> Strike_Slip
        #
        # U, NS, TS and similar codes carry no usable type
        # =========================================================

        regimeType = {
            'NF': 'Normal',
            'TF': 'Reverse',
            'SS': 'Strike_Slip'
        }


        # =========================================================
        # 10. Azimuth difference function
        #
        # azi describes an axis:
        #
        # 0 deg and 180 deg are equivalent
        #
        # therefore the final difference always lies in:
        #
        # 0° <= angle <= 90°
        # =========================================================

        def aziAngleDifference(aziA, aziB):

            aziA = float(aziA)
            aziB = float(aziB)

            # First restrict the values to 0~180
            aziA = aziA % 180.0
            aziB = aziB % 180.0

            difference = abs(
                aziA - aziB
            )

            # The axial difference is limited to 0~90 deg
            difference = min(
                difference,
                180.0 - difference
            )

            return difference


        # =========================================================
        # 11. Longitude/latitude distance function
        #
        # Spherical Haversine distance
        #
        # unit: km
        # =========================================================

        def haversineDistance(
                lat1,
                lon1,
                lat2,
                lon2
            ):

            earthRadius = 6371.0

            lat1 = np.deg2rad(lat1)
            lon1 = np.deg2rad(lon1)

            lat2 = np.deg2rad(lat2)
            lon2 = np.deg2rad(lon2)


            deltaLat = (
                lat2 - lat1
            )

            deltaLon = (
                lon2 - lon1
            )


            a = (
                np.sin(deltaLat / 2.0)**2
                +
                np.cos(lat1)
                * np.cos(lat2)
                * np.sin(deltaLon / 2.0)**2
            )


            c = 2.0 * np.arcsin(
                np.sqrt(a)
            )


            return (
                earthRadius * c
            )


        # =========================================================
        # 12. Convert the database to numpy arrays
        # =========================================================

        databaseLat = (
            data['LAT']
            .to_numpy(dtype=float)
        )

        databaseLon = (
            data['LON']
            .to_numpy(dtype=float)
        )

        databaseAzi = (
            data['AZI']
            .to_numpy(dtype=float)
        )

        databaseRegime = (
            data['REGIME']
            .to_numpy()
        )


        # =========================================================
        # 13. Check that the input is a 2-D array
        # =========================================================

        inputData = np.asarray(
            inputData,
            dtype=object
        )


        if inputData.ndim != 2:

            raise ValueError(
                "inputData must be a two-dimensional array."
            )


        if inputData.shape[1] != 6:

            raise ValueError(
                "Each row of inputData must contain 6 values: "
                "[azi1, aziType1, azi2, aziType2, lat, lon]"
            )


        # =========================================================
        # 14. Container for the final result
        # =========================================================

        result = []


        # =========================================================
        # 15. Match every fault
        # =========================================================

        for row in inputData:


            # -----------------------------------------------------
            # The two candidate solutions of the current fault
            # -----------------------------------------------------

            azi1 = float(row[0])
            aziType1 = str(row[1])

            azi2 = float(row[2])
            aziType2 = str(row[3])


            # -----------------------------------------------------
            # Coordinates of the current fault
            # -----------------------------------------------------

            faultLat = float(row[4])
            faultLon = float(row[5])


            # =====================================================
            # 16. Distance from the current fault to all database points
            # =====================================================

            distances = haversineDistance(

                faultLat,
                faultLon,

                databaseLat,
                databaseLon
            )


            # -----------------------------------------------------
            # Nearest database point
            # -----------------------------------------------------

            nearestIndex = np.argmin(
                distances
            )


            # -----------------------------------------------------
            # Information of the nearest point
            # -----------------------------------------------------

            referenceAzi = (
                databaseAzi[nearestIndex]
            )

            referenceRegime = (
                databaseRegime[nearestIndex]
            )


            # -----------------------------------------------------
            # Database REGIME mapped to a stress type
            # -----------------------------------------------------

            referenceType = regimeType.get(
                referenceRegime,
                None
            )


            # =====================================================
            # 17. First priority:
            #
            # if the database has a valid REGIME, match the candidates by type;
            # the candidate whose aziType agrees with the database wins.
            # =====================================================

            if referenceType is not None:


                type1Match = (
                    aziType1 == referenceType
                )

                type2Match = (
                    aziType2 == referenceType
                )


                # -------------------------------------------------
                # Only azi1 has the matching type
                # -------------------------------------------------

                if type1Match and not type2Match:

                    selectedAzi = azi1
                    selectedType = aziType1
                    choose = 0


                # -------------------------------------------------
                # Only azi2 has the matching type
                # -------------------------------------------------

                elif type2Match and not type1Match:

                    selectedAzi = azi2
                    selectedType = aziType2
                    choose = 1


                # -------------------------------------------------
                # Both types match
                #
                # pick the candidate with the smaller AZI difference
                # -------------------------------------------------

                elif type1Match and type2Match:

                    angle1 = aziAngleDifference(
                        azi1,
                        referenceAzi
                    )

                    angle2 = aziAngleDifference(
                        azi2,
                        referenceAzi
                    )


                    if angle1 <= angle2:

                        selectedAzi = azi1
                        selectedType = aziType1
                        choose = 0

                    else:

                        selectedAzi = azi2
                        selectedType = aziType2
                        choose = 1


                # -------------------------------------------------
                # Neither type matches
                #
                # fall back to the database AZI
                # -------------------------------------------------

                else:

                    angle1 = aziAngleDifference(
                        azi1,
                        referenceAzi
                    )

                    angle2 = aziAngleDifference(
                        azi2,
                        referenceAzi
                    )


                    if angle1 <= angle2:

                        selectedAzi = azi1
                        selectedType = aziType1
                        choose = 0

                    else:

                        selectedAzi = azi2
                        selectedType = aziType2
                        choose = 1


            # =====================================================
            # 18. The database has no valid REGIME
            #
            # (U / NS / TS / NaN and so on)
            #
            # choose the candidate with the smaller AZI difference
            # =====================================================

            else:

                angle1 = aziAngleDifference(
                    azi1,
                    referenceAzi
                )

                angle2 = aziAngleDifference(
                    azi2,
                    referenceAzi
                )


                if angle1 <= angle2:

                    selectedAzi = azi1
                    selectedType = aziType1
                    choose = 0

                else:

                    selectedAzi = azi2
                    selectedType = aziType2
                    choose = 1


            # =====================================================
            # 19. Store the result of the current fault
            # =====================================================

            result.append([
                selectedAzi,
                selectedType,
                choose
            ])


        # =========================================================
        # 20. Convert the list into an n x 3 array
        # =========================================================

        result = np.asarray(
            result,
            dtype=object
        )


        return result

class picture(geology):
    """Figure factory: stress maps, beach balls and activation risk maps.

    The class keeps the plotting defaults (line length, colour map, DPI,
    font sizes ...) and provides the renderers used by ``main.py``.
    """

    def __init__(self,path = './', lineLength = 0.5, color = 'jet', figureName = "", lineWidth = 1, 
                 pointSize = 20, dpi = 600, rlabel_fontsize = 25, thetalabel_fontsize = 25,cbar_labelsize = 25):
        super().__init__()
        self.__path = path
        self.__lineLength = lineLength
        self.__color = color
        self.__figureName = figureName
        self.__lineWidth = lineWidth
        self.__pointSize = pointSize
        self.__dpi = dpi
        self.__rlabel_fontsize = rlabel_fontsize
        self.__thetalabel_fontsize = thetalabel_fontsize
        self.__cbar_labelsize = cbar_labelsize
        # Use a serif font so that the figures match typical journal style
        plt.rcParams['font.family'] = 'Times New Roman'
        
    def setPath(self,path):
        """Set the output directory used by :meth:`save`."""
        self.__path = path
        
    def setLineLength(self,lineLength):
        """Set the length of the stress direction segments."""
        self.__lineLength = lineLength
        
    def setColor(self,color):
        """Set the colour map used by the plots."""
        self.__color = color
        
    def setFigureName(self,figureName):
        """Set the figure title (suptitle)."""
        self.__figureName = figureName
        
    def setLineWidth(self,lineWidth):
        """Set the line width used by the plots."""
        self.__lineWidth = lineWidth
        
    def setPointSize(self,pointSize):
        """Set the marker size used by the plots."""
        self.__pointSize = pointSize
        
    def setDpi(self,dpi):
        """Set the resolution of the produced figures."""
        self.__dpi = dpi
        
    def setRlabel_fontsize(self,rlabel_fontsize):
        """Set the font size of the radial labels."""
        self.__rlabel_fontsize = rlabel_fontsize
        
    def setThetalabel_fontsize(self,thetalabel_fontsize):
        """Set the font size of the angular labels."""
        self.__thetalabel_fontsize = thetalabel_fontsize
        
    def setCbar_labelsize(self,cbar_labelsize):
        """Set the font size of the colorbar labels."""
        self.__cbar_labelsize = cbar_labelsize
        
    def getDpi(self):
        """Return the current figure resolution."""
        return self.__dpi
        
    def save(self, fig, pictureFormat = '.png', pictureName = 'picture'):
        """Save *fig* into ``self.__path`` with the given format and name."""
        fig.savefig(os.path.join(self.__path, pictureName + pictureFormat), bbox_inches='tight', pad_inches=0.15)
        
    def polar_to_cartesian(self,r, theta):
        """Convert polar (r, theta) to Cartesian (x, y).

        theta is measured clockwise from north, which is the convention
        used for azimuths.
        """
        # Clockwise rotation with north as the reference direction
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return x, y

    def cartesian_to_polar(self,x, y):
        """Convert Cartesian (x, y) to polar (r, theta) with north as reference."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)  # angle measured from north
        return r, theta
    
    def vectorAddition(self,vectors):
        """Add several polar vectors (theta, r) and return (theta, r)."""
        # Accumulator in the Cartesian frame
        total_x, total_y = 0, 0
        
        for theta, r in vectors:
            # Convert each polar vector to Cartesian before summing
            x, y = self.polar_to_cartesian(r, theta)
            #print(x,y)
            total_x += x
            total_y += y
        #print(total_x,total_y)
        
        # Convert the sum back to polar coordinates
        total_r, total_theta = self.cartesian_to_polar(total_x, total_y)
        return total_theta, total_r
    
    def calculate_circle_from_three_points(self,p1, p2, p3):
        """
        Compute the centre and the radius of the circle through three points.

        The points are given as polar (theta, r) pairs.
        """
        p1 = self.polar_to_cartesian(p1[1], p1[0])
        p2 = self.polar_to_cartesian(p2[1], p2[0])
        p3 = self.polar_to_cartesian(p3[1], p3[0])
        # Linear system for the circle centre (x0, y0)
        A = np.array([
        [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
        [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])]
        ])
        b = np.array([
        [p2[0]**2 - p1[0]**2 + p2[1]**2 - p1[1]**2],
        [p3[0]**2 - p1[0]**2 + p3[1]**2 - p1[1]**2]
        ])
        # Solve for the centre
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = x0[0], y0[0]
        # Radius = distance between the centre and the first point
        r = np.sqrt((x0 - p1[0])**2 + (y0 - p1[1])**2)
        return (x0, y0), r
    
    def line_circle_intersection(self,circle_center, r, rakePoint):
        """
        Intersections of the line through the pole with a circle.

        :param circle_center: circle centre (x0, y0)
        :param r: circle radius
        :param rakePoint: polar (theta, r) point defining the line direction
        :return: the two intersection points in polar coordinates
        """
        x0, y0 = circle_center
        xr, yr = self.polar_to_cartesian(rakePoint[1], rakePoint[0])
        # Line equation: y = k * x with k = tan(theta)

        # Substituting into (x - x0)^2 + (y - y0)^2 = r^2 gives the
        # quadratic equation a * x^2 + b * x + c = 0
        a = 1 + (yr / xr)**2
        b = -2 * x0 - 2 * yr * y0 / xr
        c = y0**2 + x0**2 - r**2

        # Solve the quadratic equation
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None  # no intersection

        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        y1 = yr / xr * x1
        y2 = yr / xr * x2
        p1 = self.cartesian_to_polar(x1, y1)
        p2 = self.cartesian_to_polar(x2, y2)
        return p1,p2
    
    def stressMapPlus(
        self,
        lon,
        lat,
        azi,
        aziType,
        background,
        area,
        isTitle=False,
        boundaryPoints=None,
        boundaryColor='yellow',
        boundarySize=8,
        boundaryLine=False,
        boundaryLineWidth=0.8
        ):
        """
        Draw a stress direction map.

        Normal      : red open circle
        Reverse     : blue open square, one edge parallel to the segment
        Strike_Slip : green open triangle, apex along the segment

        Parameters
        ----------
        lon, lat : array-like
            Longitudes and latitudes of the data points.
        azi : array-like
            Stress azimuth in degrees.
        aziType : array-like
            'Normal', 'Reverse' or 'Strike_Slip'.
        background : str
            Path of the background image.
        area : tuple
            (minLon, maxLon, minLat, maxLat).
        isTitle : bool
            Whether to show the title.
        """

        # =========================================================
        # 1. Check the input
        # =========================================================

        lon = np.asarray(lon)
        lat = np.asarray(lat)
        azi = np.asarray(azi)
        aziType = np.asarray(aziType)

        if not (
            len(lon) == len(lat) ==
            len(azi) == len(aziType)
        ):
            raise ValueError(
                "lon, lat, azi and aziType must have the same length"
            )

        minLon, maxLon, minLat, maxLat = area

        if minLon >= maxLon or minLat >= maxLat:
            raise ValueError("The longitude/latitude range of area is invalid")

        # =========================================================
        # 2. Read the background image
        # =========================================================

        image = mpimg.imread(background)

        height, width = image.shape[:2]

        fig, ax = plt.subplots(
            1, 1,
            dpi=self.__dpi
        )

        ax.imshow(
            image,
            origin='upper'
        )

        # =========================================================
        # 3. Colour of every stress type
        # =========================================================

        colorDict = {
            'Normal': 'red',
            'Reverse': 'blue',
            'Strike_Slip': 'green'
        }

        # =========================================================
        # 4. Store the segments and the centre symbols
        # =========================================================

        segments = []
        colors = []
        markerData = []

        # =========================================================
        # 5. Compute the stress direction segments
        # =========================================================

        for number, (currentLon, currentLat) in tqdm.tqdm(
            enumerate(zip(lon, lat)),
            total=len(lon),
            desc='drawing'
        ):

            if not (
                np.isfinite(currentLon)
                and np.isfinite(currentLat)
                and np.isfinite(azi[number])
            ):
                continue

            # Skip the data points outside the map area
            if not (
                minLon <= currentLon <= maxLon
                and minLat <= currentLat <= maxLat
            ):
                continue

            currentType = aziType[number]

            if currentType not in colorDict:
                raise ValueError(
                    f"Unknown aziType: {currentType}"
                )

            color = colorDict[currentType]

            angle = np.deg2rad(azi[number])

            # ---------------------------------------------
            # End points of the stress direction segment
            # ---------------------------------------------

            lon1 = (
                currentLon
                - self.__lineLength / 2 * np.sin(angle)
            )

            lat1 = (
                currentLat
                - self.__lineLength / 2 * np.cos(angle)
            )

            lon2 = (
                currentLon
                + self.__lineLength / 2 * np.sin(angle)
            )

            lat2 = (
                currentLat
                + self.__lineLength / 2 * np.cos(angle)
            )

            # ---------------------------------------------
            # Convert longitude/latitude to image coordinates
            # ---------------------------------------------

            point1 = self.locTrans(
                (lon1, lat1),
                width,
                height,
                area
            )

            point2 = self.locTrans(
                (lon2, lat2),
                width,
                height,
                area
            )

            segments.append([
                point1,
                point2
            ])

            colors.append(color)

            # ---------------------------------------------
            # Coordinates of the centre symbol
            # ---------------------------------------------

            x, y = self.locTrans(
                (currentLon, currentLat),
                width,
                height,
                area
            )

            markerData.append((
                x,
                y,
                point1,
                point2,
                currentType,
                color
            ))

        # =========================================================
        # 6. Draw the stress direction segments
        # =========================================================

        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=self.__lineWidth,
            clip_on=True,
            zorder=3
        )

        ax.add_collection(lc)

        # =========================================================
        # 7. Restrict the map extent
        # =========================================================

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # =========================================================
        # 8. Set up the longitude/latitude axes
        # =========================================================

        numTicks = 5

        lonTicks = np.linspace(
            minLon,
            maxLon,
            numTicks
        )

        latTicks = np.linspace(
            minLat,
            maxLat,
            numTicks
        )

        xTicks = (
            (lonTicks - minLon)
            / (maxLon - minLon)
            * width
        )

        yTicks = (
            (maxLat - latTicks)
            / (maxLat - minLat)
            * height
        )

        ax.set_xticks(xTicks)
        ax.set_yticks(yTicks)

        ax.set_xticklabels([
            f'{value:.3f}°E'
            for value in lonTicks
        ])

        ax.set_yticklabels([
            f'{value:.3f}°N'
            for value in latTicks
        ])

        ax.tick_params(
            axis='both',
            which='both',
            direction='out',
            top=True,
            bottom=True,
            left=True,
            right=True,
            labeltop=True,
            labelbottom=True,
            labelleft=True,
            labelright=True
        )

        # =========================================================
        # 9. Title
        # =========================================================

        if isTitle:
            ax.set_title('Stress Map')

        # =========================================================
        # 10. Freeze the coordinate transform
        # =========================================================

        fig.canvas.draw()

        # =========================================================
        # 11. Draw the rotated centre symbols
        # =========================================================

        for (
            x, y,
            point1, point2,
            currentType, color
        ) in markerData:

            # ---------------------------------------------
            # Convert the segment end points to display coordinates
            # ---------------------------------------------

            p1_display = ax.transData.transform(point1)
            p2_display = ax.transData.transform(point2)

            dx = p2_display[0] - p1_display[0]
            dy = p2_display[1] - p1_display[1]

            rotation = np.rad2deg(
                np.arctan2(dy, dx)
            )

            # ---------------------------------------------
            # Normal: open circle
            # ---------------------------------------------

            if currentType == 'Normal':

                marker = MarkerStyle('o')

            # ---------------------------------------------
            # Reverse: rotated square
            # one edge stays parallel to the segment
            # ---------------------------------------------

            elif currentType == 'Reverse':

                marker = MarkerStyle(
                    's',
                    transform=Affine2D().rotate_deg(
                        rotation
                    )
                )

            # ---------------------------------------------
            # Strike-slip: rotated triangle
            # the apex points along the segment
            # ---------------------------------------------

            elif currentType == 'Strike_Slip':

                marker = MarkerStyle(
                    '^',
                    transform=Affine2D().rotate_deg(
                        rotation - 90
                    )
                )

            # ---------------------------------------------
            # Draw the centre symbol
            # ---------------------------------------------

            ax.scatter(
                x,
                y,
                s=self.__pointSize,
                marker=marker,
                facecolors='none',
                edgecolors=color,
                linewidths=self.__lineWidth,
                clip_on=True,
                zorder=5
            )
            
        # =========================================================
        # Draw the discrete boundary points
        # =========================================================

        if boundaryPoints is not None:

            boundaryPoints = np.asarray(
                boundaryPoints,
                dtype=float
            )

            if (
                boundaryPoints.ndim != 2
                or boundaryPoints.shape[1] != 2
            ):
                raise ValueError(
                    "boundaryPoints must be an N x 2 array; "
                    "each row is [longitude, latitude]"
                )

            boundaryX = []
            boundaryY = []

            for currentLon, currentLat in boundaryPoints:

                if not (
                    np.isfinite(currentLon)
                    and np.isfinite(currentLat)
                ):
                    continue

                # Convert longitude/latitude to background image pixels
                x, y = self.locTrans(
                    (currentLon, currentLat),
                    width,
                    height,
                    area
                )

                boundaryX.append(x)
                boundaryY.append(y)

            # Optional: connect the points in input order
            if boundaryLine and len(boundaryX) >= 2:

                ax.plot(
                    boundaryX,
                    boundaryY,
                    color=boundaryColor,
                    linewidth=1.2,
                    linestyle='-',
                    clip_on=True,
                    zorder=7
                )

            # Draw the discrete points
            ax.scatter(
                boundaryX,
                boundaryY,
                s=boundarySize,
                marker='x',
                c=boundaryColor,
                linewidths = boundaryLineWidth,
                clip_on=True,
                zorder=8
            )

        # =========================================================
        # 12. Legend: the marker orientation follows the horizontal line
        # =========================================================

        legend_elements = [

            # Normal: open circle
            Line2D(
                [0], [0],
                color='red',
                linewidth=1.5,
                marker='o',
                markerfacecolor='none',
                markeredgecolor='red',
                markersize=5,
                label='Normal'
            ),

            # Reverse: open square
            Line2D(
                [0], [0],
                color='blue',
                linewidth=1.5,
                marker='s',
                markerfacecolor='none',
                markeredgecolor='blue',
                markersize=5,
                label='Reverse'
            ),

            # Strike-slip: triangle rotated by 90 deg, apex to the right
            Line2D(
                [0], [0],
                color='green',
                linewidth=1.5,

                marker=MarkerStyle(
                    '^',
                    transform=Affine2D().rotate_deg(-90)
                ),

                markerfacecolor='none',
                markeredgecolor='green',
                markersize=6,
                label='Strike-slip'
            ),
            
            #Boundary
            Line2D(
                [0], [0],
                color='none',
                linestyle='None',
                marker='x',
                markeredgecolor=boundaryColor,
                markeredgewidth=0.4,
                markersize=4,
                label='Boundary'
            )
        ]

        # =========================================================
        # Legend position
        # =========================================================

        ax.legend(
            handles=legend_elements,

            # top left corner inside the map
            loc='upper left',
            bbox_to_anchor=(0.008, 0.992),

            ncol=1,

            # smaller font and symbols
            fontsize=6,
            markerscale=0.75,

            # smaller legend lines and spacing
            handlelength=1.2,
            handleheight=0.6,
            handletextpad=0.35,
            labelspacing=0.20,

            # smaller legend border padding
            borderpad=0.25,
            borderaxespad=0.20,

            frameon=True,
            facecolor='#555555',   # dark grey background
            edgecolor='none',      # no border
            framealpha=1.0,        # fully opaque, EPS friendly
            labelcolor='white'     # white legend text
            )

        return fig
    
    def FaultsActivationRiskMap(
            self,
            eigvecs,
            Phi,
            focalData,
            background,
            area,
            isTitle=False,
            smoothing=0.5,
            numTicks=5,
            colorbarPad=0.15,
            terrainStrength=0.45,
            saturationScale=1.2,
            boundaryPoints=None,
            boundaryColor="yellow",
            boundarySize=15,
            boundaryLine=False,
            boundaryLineWidth=0.8):
            """
            Compute the fault activation risk and draw the risk map.

            Parameters
            ----------
            eigvecs : array-like
                Principal stress eigenvectors of every data point.

            Phi : array-like
                Phi value of every data point.

            focalData : ndarray
                focalData[:, 0] : strike
                focalData[:, 1] : dip
                focalData[:, 2] : slip
                focalData[:, 3] : latitude
                focalData[:, 4] : longitude

            background : str
                Path of the background image.

            area : tuple
                Drawing window:
                (minLon, maxLon, minLat, maxLat)

            isTitle : bool
                Whether to show the title.

            smoothing : float
                RBF smoothing coefficient.
                Larger values give a smoother field, 0.1 ~ 2.0 recommended.
                Default 0.5.

            numTicks : int
                Number of ticks on each axis.
                Default 5.

            colorbarPad : float
                Distance between the colorbar and the main map.
                Larger values move the colorbar further to the right.
                Default 0.15.

            terrainStrength : float
                Strength of the terrain shading, suggested range 0 ~ 1.

                0   : almost no terrain shading
                0.3 : weak
                0.45: recommended
                0.6 : strong
                1.0 : fully use the brightness of the original background

            saturationScale : float
                Saturation boost applied to the risk colours.
                Default 1.2.

            boundaryPoints : array-like, optional
                Boundary points of shape (N, 2), each row [longitude, latitude].
                When None neither the boundary nor its legend is drawn.

            boundaryColor : color, default "yellow"
                Colour of the boundary points and of the optional line.

            boundarySize : float, default 15
                Marker area of the x markers, in pt^2.

            boundaryLine : bool, default False
                Whether to connect the boundary points in input order.

            boundaryLineWidth : float, default 0.4
                Line width of the x markers and of the optional connecting line, in pt.

            Returns
            -------
            fig
                matplotlib Figure.

            risk
                Risk value of every valid data point.
            """

            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            import matplotlib.colors as mcolors

            from scipy.interpolate import RBFInterpolator
            from matplotlib.ticker import FuncFormatter
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable


            # =========================================================
            # 1. Read focalData
            # =========================================================

            strike = focalData[:, 0]
            dip = focalData[:, 1]
            slip = focalData[:, 2]

            lat = focalData[:, 3]
            lon = focalData[:, 4]


            # =========================================================
            # 2. Basic input checks
            # =========================================================

            numberData = len(focalData)

            if len(eigvecs) != numberData:

                raise ValueError(
                    "eigvecs and focalData do not contain the same number of data points."
                )

            if len(Phi) != numberData:

                raise ValueError(
                    "Phi and focalData do not contain the same number of data points."
                )


            # =========================================================
            # 3. Compute the Fault Activation Risk
            #
            # Risk = ||Shear Stress|| / ||Normal Stress||
            # =========================================================

            risk = np.zeros(
                numberData,
                dtype=float
            )


            for num, eigvec in enumerate(eigvecs):


                # -----------------------------------------------------
                # Set the fault geometry
                # -----------------------------------------------------

                self.setFaultInDeg([
                    strike[num],
                    dip[num],
                    slip[num]
                ])


                # -----------------------------------------------------
                # Fault normal vector
                # -----------------------------------------------------

                normalVector = self.faultNormalVector()


                # -----------------------------------------------------
                # k1
                # -----------------------------------------------------

                denominator = np.sqrt(
                    2 * Phi[num]**2
                    - 2 * Phi[num]
                    + 2
                )


                k1 = (
                    Phi[num] + 1
                ) / denominator


                # -----------------------------------------------------
                # k2
                # -----------------------------------------------------

                k2 = (
                    (2 - Phi[num])
                    + (Phi[num] + 1)
                ) / denominator


                # -----------------------------------------------------
                # Direction cosines
                # -----------------------------------------------------

                cosine1 = np.dot(
                    normalVector,
                    eigvec[:, 0]
                )

                cosine2 = np.dot(
                    normalVector,
                    eigvec[:, 1]
                )

                cosine3 = np.dot(
                    normalVector,
                    eigvec[:, 2]
                )


                # -----------------------------------------------------
                # Normal Stress Vector
                # -----------------------------------------------------

                normalStress = np.array([
                    cosine1,
                    cosine2,
                    cosine3
                ]) * (
                    k1
                    + k2 * (
                        cosine1**2
                        + Phi[num] * cosine2**2
                    )
                )


                # -----------------------------------------------------
                # Shear Stress Vector
                # -----------------------------------------------------

                shearStress = np.array([

                    (
                        1
                        - cosine1**2
                        - Phi[num] * cosine2**2
                    ) * cosine1,

                    (
                        Phi[num]
                        - cosine1**2
                        - Phi[num] * cosine2**2
                    ) * cosine2,

                    (
                        -cosine1**2
                        - Phi[num] * cosine2**2
                    ) * cosine3

                ]) * k2


                # -----------------------------------------------------
                # Magnitudes
                # -----------------------------------------------------

                normalMagnitude = np.linalg.norm(
                    normalStress
                )

                shearMagnitude = np.linalg.norm(
                    shearStress
                )


                # -----------------------------------------------------
                # Risk
                # -----------------------------------------------------

                if normalMagnitude > 0:

                    risk[num] = (
                        shearMagnitude
                        / normalMagnitude
                    )

                else:

                    risk[num] = np.nan


            # =========================================================
            # 4. Drop the invalid data
            #
            # Note:
            # the data is NOT filtered by area!
            #
            # Valid points outside area still take part in the RBF interpolation,
            # which keeps the interpolation near the border sensible.
            # =========================================================

            validMask = (
                np.isfinite(lon)
                &
                np.isfinite(lat)
                &
                np.isfinite(risk)
            )


            lonValid = lon[validMask]
            latValid = lat[validMask]
            riskValid = risk[validMask]


            if len(riskValid) < 3:

                raise ValueError(
                    "Too few valid data points for the spatial interpolation."
                )


            # =========================================================
            # 5. Read the background image
            # =========================================================

            image = mpimg.imread(
                background
            )


            # ---------------------------------------------------------
            # Grayscale -> RGB
            # ---------------------------------------------------------

            if image.ndim == 2:

                image = np.stack(
                    [
                        image,
                        image,
                        image
                    ],
                    axis=-1
                )


            # ---------------------------------------------------------
            # RGBA -> RGB
            # ---------------------------------------------------------

            if image.shape[2] == 4:

                image = image[:, :, :3]


            # ---------------------------------------------------------
            # Convert to float
            # ---------------------------------------------------------

            image = image.astype(float)


            # ---------------------------------------------------------
            # Scale RGB to 0 ~ 1
            # ---------------------------------------------------------

            if image.max() > 1.0:

                image = (
                    image / 255.0
                )


            image = np.clip(
                image,
                0,
                1
            )


            imageHeight, imageWidth = (
                image.shape[:2]
            )


            # =========================================================
            # 6. Final drawing area
            # =========================================================

            minLon, maxLon, minLat, maxLat = area


            if minLon >= maxLon:

                raise ValueError(
                    "area must satisfy minLon < maxLon."
                )

            if minLat >= maxLat:

                raise ValueError(
                    "area must satisfy minLat < maxLat."
                )


            # =========================================================
            # 7. Build a lon/lat grid that matches the background pixels one to one
            #
            # The image:
            #
            # top left     = (minLon, maxLat)
            # bottom right = (maxLon, minLat)
            #
            # =========================================================

            gridLon1D = np.linspace(
                minLon,
                maxLon,
                imageWidth
            )


            # ---------------------------------------------------------
            # Mind the latitude direction
            #
            # first image row -> maxLat
            # last image row  -> minLat
            # ---------------------------------------------------------

            gridLat1D = np.linspace(
                maxLat,
                minLat,
                imageHeight
            )


            gridLon, gridLat = np.meshgrid(
                gridLon1D,
                gridLat1D
            )


            # =========================================================
            # 8. Build the RBF interpolator
            #
            # All valid points take part in the interpolation,
            # including the points outside area.
            # =========================================================

            points = np.column_stack([
                lonValid,
                latValid
            ])


            rbf = RBFInterpolator(

                points,

                riskValid,

                kernel='thin_plate_spline',

                smoothing=smoothing
            )


            # =========================================================
            # 9. Evaluate Risk inside area
            #
            # Only the pixels inside area are evaluated,
            # but the interpolator itself is built from all the data.
            # =========================================================

            gridPoints = np.column_stack([

                gridLon.ravel(),

                gridLat.ravel()

            ])


            gridRisk = rbf(
                gridPoints
            )


            gridRisk = gridRisk.reshape(
                imageHeight,
                imageWidth
            )


            # =========================================================
            # 10. Colour scale range of Risk
            #
            # Use the actual Risk range of the displayed area.
            # =========================================================

            riskMin = np.nanmin(
                gridRisk
            )

            riskMax = np.nanmax(
                gridRisk
            )


            if not np.isfinite(riskMin):

                raise ValueError(
                    "The interpolation result contains no valid Risk value."
                )


            # =========================================================
            # 11. Normalise Risk
            # =========================================================

            if np.isclose(
                riskMax,
                riskMin
            ):

                riskNormalized = np.zeros_like(
                    gridRisk
                )

            else:

                riskNormalized = (
                    gridRisk - riskMin
                ) / (
                    riskMax - riskMin
                )


            riskNormalized = np.clip(
                riskNormalized,
                0,
                1
            )


            # =========================================================
            # 12. Use the colour map defined on the class
            #
            # set by the user:
            #
            # cmap = self.__color
            # =========================================================

            cmap = self.__color
            cmapObject = plt.get_cmap(cmap)


            # =========================================================
            # 13. Risk -> RGB
            # =========================================================

            riskRGBA = cmapObject(riskNormalized)
            riskRGB = riskRGBA[:, :, :3]


            # =========================================================
            # 14. Risk RGB -> HSV
            #
            # Risk provides:
            #
            # H = Hue
            # S = Saturation
            #
            # =========================================================

            riskHSV = mcolors.rgb_to_hsv(
                riskRGB
            )


            # =========================================================
            # 15. Background RGB -> HSV
            #
            # The background mainly provides the terrain brightness.
            # =========================================================

            backgroundHSV = mcolors.rgb_to_hsv(
                image
            )


            terrainValue = (
                backgroundHSV[:, :, 2]
            )


            # =========================================================
            # 16. Adjust the terrain brightness
            #
            # original:
            #
            # terrainValue = 0 ~ 1
            #
            # transformed:
            #
            # Vnew =
            # (1 - terrainStrength)
            # +
            # terrainStrength * Vold
            #
            #
            # terrainStrength = 0:
            # no terrain shading at all
            #
            # terrainStrength = 1:
            # fully use the original background brightness
            #
            # default 0.45:
            # the relief stays visible while the Risk colours stay bright.
            # =========================================================

            terrainStrength = np.clip(
                terrainStrength,
                0,
                1
            )


            terrainValue = (
                (1.0 - terrainStrength)
                +
                terrainStrength * terrainValue
            )


            # =========================================================
            # 17. Create the final HSV image
            # =========================================================

            resultHSV = np.zeros_like(
                riskHSV
            )


            # ---------------------------------------------------------
            # Hue comes from Risk
            # ---------------------------------------------------------

            resultHSV[:, :, 0] = (
                riskHSV[:, :, 0]
            )


            # ---------------------------------------------------------
            # Saturation comes from Risk
            #
            # Slightly boost the saturation so that
            # the grey terrain does not wash out the colours.
            # ---------------------------------------------------------

            resultHSV[:, :, 1] = np.clip(

                riskHSV[:, :, 1]
                * saturationScale,

                0,
                1
            )


            # ---------------------------------------------------------
            # Value comes from the terrain
            # ---------------------------------------------------------

            resultHSV[:, :, 2] = (
                terrainValue
            )


            # =========================================================
            # 18. HSV -> RGB
            #
            # final result:
            #
            # Risk    -> colour
            # Terrain -> brightness
            #
            # no transparency is used.
            # =========================================================

            resultImage = mcolors.hsv_to_rgb(
                resultHSV
            )


            resultImage = np.clip(
                resultImage,
                0,
                1
            )


            # =========================================================
            # 19. Create the Figure
            # =========================================================

            fig, ax = plt.subplots(

                1,
                1,

                dpi=self.__dpi
            )


            # =========================================================
            # 20. Draw the fused image
            # =========================================================

            ax.imshow(

                resultImage,

                extent=[
                    minLon,
                    maxLon,
                    minLat,
                    maxLat
                ],

                origin='upper',

                aspect='auto'
            )


            # =========================================================
            # 21. Strictly limit the display range
            # =========================================================

            ax.set_xlim(
                minLon,
                maxLon
            )


            ax.set_ylim(
                minLat,
                maxLat
            )


            # =========================================================
            # 22. Use a sparse set of longitude/latitude ticks
            # =========================================================

            lonTicks = np.linspace(
                minLon,
                maxLon,
                numTicks
            )


            latTicks = np.linspace(
                minLat,
                maxLat,
                numTicks
            )


            ax.set_xticks(
                lonTicks
            )


            ax.set_yticks(
                latTicks
            )


            # =========================================================
            # 23. Longitude/latitude format
            # =========================================================

            def longitudeFormatter(
                    value,
                    position
                ):

                if value > 0:

                    return (
                        f'{abs(value):.3f}°E'
                    )

                elif value < 0:

                    return (
                        f'{abs(value):.3f}°W'
                    )

                else:

                    return '0.000°'


            def latitudeFormatter(
                    value,
                    position
                ):

                if value > 0:

                    return (
                        f'{abs(value):.3f}°N'
                    )

                elif value < 0:

                    return (
                        f'{abs(value):.3f}°S'
                    )

                else:

                    return '0.000°'


            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    longitudeFormatter
                )
            )


            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    latitudeFormatter
                )
            )


            # =========================================================
            # 24. Show the longitude on both sides
            # =========================================================

            ax.tick_params(

                axis='x',

                top=True,
                bottom=True,

                labeltop=True,
                labelbottom=True,

                direction='out'
            )


            # =========================================================
            # 25. Show the latitude on both sides
            # =========================================================

            ax.tick_params(

                axis='y',

                left=True,
                right=True,

                labelleft=True,
                labelright=True,

                direction='out'
            )


            # =========================================================
            # 26. Colorbar
            #
            # The colorbar must match the real Risk values,
            # not the normalised 0~1 range.
            # =========================================================

            norm = Normalize(vmin=riskMin,vmax=riskMax)
            sm = ScalarMappable(
                norm=norm,
                cmap=cmapObject
            )

            sm.set_array([])


            # =========================================================
            # 27. Add the colorbar
            #
            # colorbarPad controls how far it moves to the right.
            # =========================================================

            cbar = fig.colorbar(

                sm,

                ax=ax,

                pad=colorbarPad,

                fraction=0.046
            )


            cbar.set_label(
                'Fault Activation Risk'
            )


            # =========================================================
            # 28. Draw the boundary points and their legend
            # the axes are lon/lat, so locTrans is not needed here
            # =========================================================

            if boundaryPoints is not None:
                from matplotlib.lines import Line2D

                boundaryArray = np.asarray(boundaryPoints, dtype=float)
                if boundaryArray.ndim != 2 or boundaryArray.shape[1] != 2:
                    raise ValueError(
                        "boundaryPoints must be an N x 2 array; each row is [longitude, latitude]."
                    )

                # [nan, nan] entries separate different boundary segments
                finiteMask = np.isfinite(boundaryArray).all(axis=1)
                visibleMask = (
                    finiteMask
                    & (boundaryArray[:, 0] >= minLon)
                    & (boundaryArray[:, 0] <= maxLon)
                    & (boundaryArray[:, 1] >= minLat)
                    & (boundaryArray[:, 1] <= maxLat)
                )

                if boundaryLine:
                    # keep the NaN breaks; the axes clip out-of-range segments
                    ax.plot(
                        boundaryArray[:, 0],
                        boundaryArray[:, 1],
                        color=boundaryColor,
                        linewidth=boundaryLineWidth,
                        linestyle='-',
                        clip_on=True,
                        zorder=7
                    )

                if np.any(visibleMask):
                    ax.scatter(
                        boundaryArray[visibleMask, 0],
                        boundaryArray[visibleMask, 1],
                        s=boundarySize,
                        marker='x',
                        c=boundaryColor,
                        linewidths=boundaryLineWidth,
                        clip_on=True,
                        zorder=8
                    )

                    boundaryLegend = Line2D(
                        [0], [0],
                        linestyle='None',
                        marker='x',
                        color=boundaryColor,
                        markeredgewidth=boundaryLineWidth,
                        markersize=3.5,
                        label='Boundary'
                    )
                    ax.legend(
                        handles=[boundaryLegend],
                        loc='upper left',
                        bbox_to_anchor=(0.008, 0.992),
                        fontsize=6,
                        markerscale=0.85,
                        handlelength=1.0,
                        handletextpad=0.3,
                        labelspacing=0.1,
                        borderpad=0.2,
                        borderaxespad=0.2,
                        frameon=True,
                        facecolor='#555555',
                        edgecolor='none',
                        framealpha=1.0,
                        labelcolor='white'
                    )


            # =========================================================
            # 29. Title
            # =========================================================

            if isTitle:

                ax.set_title(
                    'Fault Activation Risk Map'
                )


            # =========================================================
            # 30. Return
            # =========================================================

            return fig, risk
             
    def stressMapWithDatabase(self, lon, lat, azi, aziType,
                               background, area, csvFile,
                              isTitle=False, numTicks=5, 
                              boundaryPoints=None, boundaryColor="yellow",
                              boundarySize=15, boundaryLine=False,
                              boundaryLineWidth=0.4):
        """Draw the database (white) and the user input (black) stress directions.

        The database must contain LAT, LON, AZI, QUALITY and REGIME.
        REGIME: N/Normal, R/Reverse, SS/Strike_Slip, U/Unknown.
        The user input may specify the stress type through aziType;
        when it is omitted, Normal is assumed.
        """
        minLon, maxLon, minLat, maxLat = area
        if not (minLon < maxLon and minLat < maxLat):
            raise ValueError('area must be (minLon, maxLon, minLat, maxLat) with valid limits')
        if numTicks < 2:
            raise ValueError('numTicks must be at least 2')

        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        azi = np.asarray(azi, dtype=float).ravel()
        if not (len(lon) == len(lat) == len(azi)):
            raise ValueError('lon, lat and azi must have the same length')

        if aziType is None:
            user_types_raw = ['Normal'] * len(lon)
        elif isinstance(aziType, str):
            user_types_raw = [aziType] * len(lon)
        else:
            user_types_raw = np.asarray(aziType).ravel().tolist()
        if len(user_types_raw) != len(lon):
            raise ValueError('aziType must have the same length as lon, lat and azi')

        database = pd.read_csv(csvFile, low_memory=False)
        database.columns = database.columns.str.strip().str.upper()
        required = ['LAT', 'LON', 'AZI', 'QUALITY', 'REGIME']
        missing = [c for c in required if c not in database.columns]
        if missing:
            raise ValueError('The CSV file is missing the following columns: ' + ', '.join(missing))
        database = database[required].copy()
        for col in ['LAT', 'LON', 'AZI']:
            database[col] = pd.to_numeric(database[col], errors='coerce')
        database['QUALITY'] = database['QUALITY'].astype('string').str.strip().str.upper()
        database['REGIME'] = database['REGIME'].astype('string').str.strip().str.upper()
        database = database.dropna(subset=['LAT', 'LON', 'AZI'])
        database = database.loc[
            database['QUALITY'].isin(['A', 'B', 'C'])
            & database['LON'].between(minLon, maxLon)
            & database['LAT'].between(minLat, maxLat)
        ].copy()

        # Common REGIME abbreviations; unknown values other than U raise an
        # error so that they are not silently misclassified.
        regime_map = {
            'N': 'Normal', 'NF': 'Normal', 'NORMAL': 'Normal','NS': 'Normal',
            'R': 'Reverse', 'RF': 'Reverse', 'TF': 'Reverse', 'REVERSE': 'Reverse', 'TS': 'Reverse',
            'SS': 'Strike_Slip', 'S': 'Strike_Slip',
            'STRIKE_SLIP': 'Strike_Slip', 'STRIKE-SLIP': 'Strike_Slip',
            'STRIKE SLIP': 'Strike_Slip',
            'U': 'U', 'UNKNOWN': 'U', 'UNDETERMINED': 'U'
        }
        def normalize_type(value):
            if pd.isna(value):
                raise ValueError('The stress type cannot be a missing value')
            key = str(value).strip().upper()
            if key not in regime_map:
                raise ValueError(f'Unrecognised user stress type: {value!r}')
            return regime_map[key]

        user_types = [normalize_type(v) for v in user_types_raw]
        database['TYPE'] = database['REGIME'].map(regime_map)
        invalid = sorted(database.loc[database['TYPE'].isna(), 'REGIME'].dropna().unique().tolist())
        if invalid or database['TYPE'].isna().any():
            raise ValueError(f'Unrecognised REGIME value(s): {invalid} (missing value present: {database["TYPE"].isna().any()}). Please check the encoding.')

        image = mpimg.imread(background)
        h, w = image.shape[:2]
        fig, ax = plt.subplots(1, 1, dpi=self.__dpi)
        ax.imshow(image, origin='upper')

        # Store segments and markers separately: fix the axes first, then rotate the markers with their on-screen direction.
        db_segments, user_segments = [], []
        db_markers, user_markers = [], []

        def make_geometry(currentLon, currentLat, currentAzi):
            theta = np.deg2rad(currentAzi)
            half = self.__lineLength / 2.0
            p1 = self.locTrans((currentLon - half * np.sin(theta),
                                currentLat - half * np.cos(theta)), w, h, area)
            p2 = self.locTrans((currentLon + half * np.sin(theta),
                                currentLat + half * np.cos(theta)), w, h, area)
            center = self.locTrans((currentLon, currentLat), w, h, area)
            return p1, p2, center

        for row in database.itertuples(index=False):
            p1, p2, center = make_geometry(row.LON, row.LAT, row.AZI)
            db_segments.append([p1, p2])
            db_markers.append((center, p1, p2, row.TYPE))

        for currentLon, currentLat, currentAzi, currentType in zip(lon, lat, azi, user_types):
            if not np.all(np.isfinite([currentLon, currentLat, currentAzi])):
                continue
            if not (minLon <= currentLon <= maxLon and minLat <= currentLat <= maxLat):
                continue
            p1, p2, center = make_geometry(currentLon, currentLat, currentAzi)
            user_segments.append([p1, p2])
            user_markers.append((center, p1, p2, currentType))

        if db_segments:
            ax.add_collection(LineCollection(db_segments, colors='white',
                              linewidths=self.__lineWidth, clip_on=True, zorder=3))
        if user_segments:
            ax.add_collection(LineCollection(user_segments, colors='black',
                              linewidths=self.__lineWidth, clip_on=True, zorder=5))

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        lonTicks = np.linspace(minLon, maxLon, numTicks)
        latTicks = np.linspace(minLat, maxLat, numTicks)
        ax.set_xticks((lonTicks - minLon) / (maxLon - minLon) * w)
        ax.set_yticks((maxLat - latTicks) / (maxLat - minLat) * h)

        def longitude_label(value):
            return f'{abs(value):.3f}°' + ('E' if value > 0 else 'W' if value < 0 else '')

        def latitude_label(value):
            return f'{abs(value):.3f}°' + ('N' if value > 0 else 'S' if value < 0 else '')

        ax.set_xticklabels([longitude_label(v) for v in lonTicks])
        ax.set_yticklabels([latitude_label(v) for v in latTicks])
        ax.tick_params(axis='x', top=True, bottom=True, labeltop=True,
                       labelbottom=True, direction='out')
        ax.tick_params(axis='y', left=True, right=True, labelleft=True,
                       labelright=True, direction='out')
        if isTitle:
            ax.set_title('Stress Map')

        # Use the public MarkerStyle(transform=...) API instead of touching
        # private attributes.  The transform angle is measured in display
        # coordinates, not in longitude/latitude degrees.
        def make_marker(kind, rotation):
            if kind == 'Normal':
                return MarkerStyle('o')
            if kind == 'Reverse':
                return MarkerStyle('s', transform=Affine2D().rotate_deg(rotation))
            if kind == 'Strike_Slip':
                return MarkerStyle('^', transform=Affine2D().rotate_deg(rotation - 90))
            if kind == 'U':
                return MarkerStyle('D', transform=Affine2D().rotate_deg(rotation))
            raise ValueError(f'Unknown stress type: {kind}')

        def draw_markers(data, color, zorder):
            for center, p1, p2, kind in data:
                a, b = ax.transData.transform(np.asarray([p1, p2], dtype=float))
                angle = np.rad2deg(np.arctan2(b[1] - a[1], b[0] - a[0]))
                marker = make_marker(kind, angle)
                ax.scatter(center[0], center[1], s=self.__pointSize,
                           marker=marker, facecolors='none', edgecolors=color,
                           linewidths=self.__lineWidth, clip_on=True, zorder=zorder)

        # Draw once to freeze ax.transData; the axes are not moved afterwards.
        fig.canvas.draw()
        draw_markers(db_markers, 'white', 4)
        draw_markers(user_markers, 'black', 6)

        # The legend uses horizontal segments; every centre symbol matches
        # the shape of its type (white = WSM, black = inversion).
        legend_specs = [
            ('Normal (WSM)', 'Normal', 'white'),
            ('Reverse (WSM)', 'Reverse', 'white'),
            ('Strike-slip (WSM)', 'Strike_Slip', 'white'),
            ('Unknown (WSM)', 'U', 'white'),
            ('Normal (Inversion)', 'Normal', 'black'),
            ('Reverse (Inversion)', 'Reverse', 'black'),
            ('Strike-slip (Inversion)', 'Strike_Slip', 'black'),
        ]
        legend_handles = [
            Line2D([0], [0], color=color, linewidth=0.4,
                   marker=make_marker(kind, 0), markerfacecolor='none',
                   markeredgecolor=color, markeredgewidth=0.4,
                   markersize=3, label=label)
            for label, kind, color in legend_specs
        ]

        # boundaryPoints is an N x 2 array with columns [longitude, latitude].
        # The Boundary legend entry is added only when points are supplied.
        if boundaryPoints is not None:
            boundary = np.asarray(boundaryPoints, dtype=float)
            if boundary.ndim != 2 or boundary.shape[1] != 2:
                raise ValueError('boundaryPoints must be an N x 2 array; each row is [longitude, latitude]')

            # Keep the NaN breaks: the line does not jump across missing points; the scatter ignores NaN.
            coords = np.full((len(boundary), 2), np.nan, dtype=float)
            for i, (blon, blat) in enumerate(boundary):
                if np.isfinite(blon) and np.isfinite(blat):
                    coords[i] = self.locTrans((blon, blat), w, h, area)

            valid = np.all(np.isfinite(coords), axis=1)
            if boundaryLine and len(coords) >= 2:
                ax.plot(coords[:, 0], coords[:, 1], color=boundaryColor,
                        linewidth=boundaryLineWidth, zorder=7, clip_on=True)
            if valid.any():
                ax.scatter(coords[valid, 0], coords[valid, 1],
                           s=boundarySize, marker='x', c=boundaryColor,
                           linewidths=boundaryLineWidth, zorder=8, clip_on=True)

            legend_handles.append(
                Line2D([0], [0], linestyle='None', color=boundaryColor,
                       marker='x', markeredgecolor=boundaryColor,
                       markeredgewidth=boundaryLineWidth, markersize=4,
                       label='Boundary')
            )

        ax.legend(
            handles=legend_handles,
            loc='upper left', bbox_to_anchor=(0.008, 0.992), ncol=1,
            fontsize=4.8, markerscale=0.55,
            handlelength=1.0, handleheight=0.45,
            handletextpad=0.25, labelspacing=0.10,
            borderpad=0.15, borderaxespad=0.15,
            frameon=True, facecolor='#555555', edgecolor='none',
            framealpha=1.0, labelcolor='white'
        )

        return fig
    
    def apmMap(self,lon,lat,apm,background,area):
        """Draw the absolute plate motion vectors on a background map.

        *apm* holds the azimuth of every vector in degrees.
        """
        image = mpimg.imread(background)
        (w,l,h) = np.shape(image)
        fig,ax = plt.subplots(1,1,dpi = self.__dpi)
        ax.imshow(image)
        for number,segment in tqdm.tqdm(enumerate(zip(lon,lat)),total = len(lon),desc = 'drawing'):
            # Convert the position to image pixels, then draw the arrow
            [x,y] = self.locTrans(segment, l, w, area)
            ax.arrow(x,y,np.sin(np.deg2rad(apm[number])) * self.__lineLength,np.cos(np.deg2rad(apm[number])) * self.__lineLength,width=0.00000000001,head_width=10,color = 'blue')
        ax.set_title('APM Map')
        return fig
    
    def visualization(self, stress=((),), isShow=False, isThetaLabel=True, isSparse = False, isShowTitle = False, cbar_position = 'bottom'):
        """Draw the sigma1 / sigma3 polar plots of the accepted stress states.

        Every accepted stress tensor is drawn as a line in the sigma1 (left)
        and sigma3 (right) hemisphere; the line colour encodes the stress
        ratio Phi.
        """
        parameterX = np.array([])
        parameterZ = np.array([])
        pointZ = np.array([])
        pointX = np.array([])
        R = np.array([])
        
        # Set the default font size
        if self.__rlabel_fontsize is None:
            self.__rlabel_fontsize = plt.rcParams['axes.labelsize']  # use the current default font size
        
        for i in stress:
            [alpha,beta,gamma,phi] = i
            x = np.array([1,0,0])
            z = np.array([0,0,1])
            r = self.Rzyz(alpha, beta, gamma)
            x = np.dot(r, x)
            z = np.dot(r, z)
            strikeX = CalAngle(np.array([0,-1]), np.array([x[0],x[1]]))
            tempX = np.array([x[0],x[1],0])
            if np.linalg.norm(tempX) == 0:
                dipX = np.pi / 2
            else:
                if np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x) > 1 or np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x) < -1:
                    if np.abs(np.abs(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x)) - 1) < 0.00001:
                        dipX = np.arccos(round(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x)))
                    else:
                        raise Exception(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x))
                else:
                    dipX = np.arccos(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x))
            
            strikeZ = CalAngle(np.array([0,-1]), np.array([z[0],z[1]]))
            tempZ = np.array([z[0],z[1],0])
            if np.linalg.norm(tempZ) == 0:
                dipZ = np.pi / 2
            else:
                if np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z) > 1 or np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z) < -1:
                    if np.abs(np.abs(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)) - 1) < 0.00001:
                        dipZ = np.arccos(round(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)))
                    else:
                        raise Exception(np.abs(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)))
                else:
                    dipZ = np.arccos(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z))
            
            parameterX = np.append(parameterX, [strikeX,dipX])
            parameterZ = np.append(parameterZ, [strikeZ,dipZ])
            lengthZ = np.cos(dipZ) * self.__lineLength
            lengthX = np.cos(dipX) * self.__lineLength
            pointZ = np.append(pointZ, self.vectorAddition([(strikeX,np.pi / 2 - dipX),(strikeZ,lengthZ)]))
            pointX = np.append(pointX, self.vectorAddition([(strikeZ,np.pi / 2 - dipZ),(strikeX,lengthX)]))
            R = np.append(R, phi)
        
        parameterX = parameterX.reshape((-1,2))
        parameterZ = parameterZ.reshape((-1,2))
        
        fig = plt.figure(dpi = self.__dpi,figsize = (20,10))
        
        r_values = [90,80,70,60,50,40,30,20,10,0]
        if isThetaLabel:
            if isSparse:
                r_labels = ['90°', '', '', '60°', '', '', '30°', '', '', '0°']
            else:
                r_labels = ['90°', '80°', '70°', '60°', '50°', '40°', '30°', '20°', '10°', '0°']  
        else:
            r_labels = ['', '', '', '', '', '', '', '', '', '']
        
        fig.suptitle(self.__figureName, fontsize=24, x=0.5, y=0.9, horizontalalignment='center', va='bottom')
        pointZ = pointZ.reshape((-1,2))
        pointX = pointX.reshape((-1,2))
        ax1 = plt.subplot(121, projection = 'polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rlim(90, 0)
        ax1.tick_params(axis='both', labelsize=self.__thetalabel_fontsize)
        
        # Font size of the radial grid labels
        ax1.set_rgrids(r_values, labels=r_labels, angle=22.5, fontsize=self.__rlabel_fontsize)
        #ax1.set_xticklabels([])  # hide the angular tick labels
        
        # Colour of every segment
        segments = []
        for i in range(len(parameterX)):
            segment = [[parameterX[i,0], np.rad2deg(parameterX[i,1])], [pointZ[i,0], np.rad2deg(np.pi / 2 - pointZ[i,1])]]
            segments.append(segment)
        lc = LineCollection(segments, cmap = self.__color, linewidth = self.__lineWidth, norm = Normalize(vmin = 0, vmax = 1))
        lc.set_array(R)
        ax1.add_collection(lc)
        ax1.scatter(parameterX[:,0], np.rad2deg(parameterX[:,1]), s = self.__pointSize, c = R, cmap = self.__color, vmin = 0, vmax = 1, marker = "s", edgecolors='none')
        if isShowTitle:
            ax1.set_title('σ₁',fontsize=50)
        
        ax2 = plt.subplot(122, projection = 'polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rlim(90, 0)
        ax2.tick_params(axis='both', labelsize=self.__thetalabel_fontsize,zorder=1000)
        
        # Font size of the radial grid labels
        ax2.set_rgrids(r_values, labels=r_labels, angle=22.5, fontsize=self.__rlabel_fontsize)
        #ax2.set_xticklabels([])  # hide the angular tick labels
        
        # Colour of every segment
        segments = []
        for i in range(len(parameterZ)):
            segment = [[parameterZ[i,0], np.rad2deg(parameterZ[i,1])], [pointX[i,0], np.rad2deg(np.pi / 2 - pointX[i,1])]]
            segments.append(segment)
        lc = LineCollection(segments, cmap = self.__color, linewidth = self.__lineWidth, norm = Normalize(vmin = 0, vmax = 1))
        lc.set_array(R)
        ax2.add_collection(lc)
        ax2.scatter(parameterZ[:,0], np.rad2deg(parameterZ[:,1]), s = self.__pointSize, c = R, cmap = self.__color, vmin = 0, vmax = 1, marker = "s", edgecolors='none')
        if isShowTitle:
            ax2.set_title('σ₃',fontsize=50)
        
        sm = ScalarMappable(cmap=self.__color, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        if cbar_position == 'bottom':
            cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        else:
            cax = add_right_cax(ax2, pad=0.1, width=0.02)
            cbar = fig.colorbar(sm, cax=cax)


        legend_angles = [0, 30, 60, 90]   # the four dip angles actually used in the figure

        legend_handles = [
            StressLegendHandle(angle)
            for angle in legend_angles
        ]

        legend_labels = [
            f'{angle}°'
            for angle in legend_angles
        ]


        # ----------------------------------------------------------
        # Draw first so that the colorbar position is final
        # ----------------------------------------------------------
        fig.canvas.draw()

        # Position of the colorbar
        cbar_pos = cax.get_position()

        # Horizontal centre of the colorbar
        cbar_center_x = cbar_pos.x0 + cbar_pos.width / 2

        # Top edge of the colorbar
        cbar_top_y = cbar_pos.y1


        # ----------------------------------------------------------
        # The four entries are laid out horizontally, centred on the colorbar
        # ----------------------------------------------------------
        legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,

            handler_map={
                StressLegendHandle: HandlerStressLegend()
            },

            # bottom centre of the legend on the top centre of the colorbar
            loc='lower center',

            bbox_to_anchor=(
                cbar_center_x,
                cbar_top_y + 0.02
            ),

            bbox_transform=fig.transFigure,

            # key point: four entries in one row
            ncol=4,

            frameon=False,

            fontsize=18,

            handlelength=2.5,
            handletextpad=0.5,

            # horizontal spacing between the entries
            columnspacing=1.5,

            # drop the extra padding of the legend
            borderaxespad=0
        )

        legend.set_zorder(10000)

        # Font size of the colorbar labels
        if self.__cbar_labelsize is not None:
            cbar.ax.tick_params(labelsize=self.__cbar_labelsize)  # font size of the tick labels
            cbar.set_label(r'Stress Ratio $\phi$', size=self.__cbar_labelsize)  # font size of the colorbar label
        fig.subplots_adjust(wspace=0.5,hspace=0.1)
        
        if isShow:
            plt.show()
        return fig
        
    def beachball(self,faults):
        """Draw beach balls for a list of faults.

        All parameters must be given in radians.
        """
        if not is_2d_array(faults):
            faults = np.array([faults])
            
        # Set the default font size
        if self.__rlabel_fontsize is None:
            self.__rlabel_fontsize = plt.rcParams['axes.labelsize']  # use the current default font size
        if self.__thetalabel_fontsize is None:
            self.__thetalabel_fontsize = plt.rcParams['axes.labelsize']
            
        fig = plt.figure(dpi = self.__dpi,figsize = (20,10))
        r_values = [90,80,70,60,50,40,30,20,10,0]
        r_labels = ['90°', '80°', '70°', '60°', '50°', '40°', '30°', '20°', '10°', '0°']
        fig.suptitle(self.__figureName, fontsize=24, x=0.5, y=0.9, horizontalalignment='center', va='bottom')
        ax1 = plt.subplot(111, projection = 'polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rlim(90, 0)
        ax1.set_rgrids(r_values, labels=r_labels, angle=22.5, fontsize = self.__rlabel_fontsize)
        ax1.tick_params(axis='both', labelsize = self.__thetalabel_fontsize)
        
        for fault in faults:
            (strike,dip,rake) = fault
            point1 = np.array([strike,90])
            point2 = np.array([strike + np.pi / 2,np.rad2deg(np.pi / 2 - dip)])
            point3 = np.array([strike + np.pi,90])
            center,radius = self.calculate_circle_from_three_points(point1, point2, point3)
            intersection1,intersection2 = self.line_circle_intersection(center, radius, np.array([strike - rake,90]))
            r1,theta1 = intersection1
            r2,theta2 = intersection2
            if theta1 < 0:
                theta1 += 2 * np.pi
            if theta2 < 0:
                theta2 += 2 * np.pi
            if strike >= np.pi:
                if theta1 >= strike or theta1 <= (strike - np.pi):
                    intersection = intersection1
                else:
                    intersection = intersection2
            else:
                if theta1 >= strike and theta1 <= (strike + np.pi):
                    intersection = intersection1
                else:
                    intersection = intersection2
            pc1 = self.polar_to_cartesian(point1[1],point1[0])
            pc3 = self.polar_to_cartesian(point3[1],point3[0])
            # Start and end angle of the arc
            start_angle = np.arctan2(pc1[1] - center[1], pc1[0] - center[0])
            end_angle = np.arctan2(pc3[1] - center[1], pc3[0] - center[0])

            # Make sure the arc runs from start_angle to end_angle
            if start_angle * end_angle >= 0:
                if start_angle > end_angle:
                    start_angle,end_angle = end_angle,start_angle
            elif start_angle > 0 and end_angle < 0:
                start_angle,end_angle = end_angle,start_angle
            else:
                start_angle,end_angle = end_angle,start_angle + np.pi * 2

            # Sample the arc
            angles = np.linspace(start_angle, end_angle, 100)
            xarray = center[0] + radius * np.cos(angles)
            yarray = center[1] + radius * np.sin(angles)
            rarray,thetaarray = self.cartesian_to_polar(xarray, yarray)
            ax1.plot(thetaarray, 90 - rarray, color='blue')
            ax1.scatter(intersection[1],90 - intersection[0],color='red')
        return fig
        
    def strikeAndDip(self,stress):
        """
        Compute strike, dip and stress ratio of the principal stress axes.

        The input is a 2-D array of stress states; the function returns the
        strike/dip of sigma1 and sigma3 together with the stress ratio.
        """
        parameterX = np.array([])
        parameterZ = np.array([])
        R = np.array([])
        #print(stress)
        for i in stress:
            #print(type(i))
            #print(i)
            [alpha,beta,gamma,phi] = i
            x = np.array([1,0,0])
            z = np.array([0,0,1])
            r = self.Rzyz(alpha, beta, gamma)
            x = np.dot(r, x)
            z = np.dot(r, z)
            strikeX = CalAngle(np.array([0,-1]), np.array([x[0],x[1]]))
            if strikeX >= np.pi:
                strikeX -= np.pi
            
            tempX = np.array([x[0],x[1],0])
            if np.linalg.norm(tempX) == 0:
                dipX = np.pi / 2
            else:
                if np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x) > 1 or np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x) < -1:
                    if np.abs(np.abs(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x)) - 1) < 0.00001:
                        dipX = np.arccos(round(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x)))
                    else:
                        raise Exception(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x))
                else:
                    dipX = np.arccos(np.dot(tempX, x) / np.linalg.norm(tempX) / np.linalg.norm(x))
            if x[2] < 0:
                dipX = -dipX
            
            strikeZ = CalAngle(np.array([0,-1]), np.array([z[0],z[1]]))
            if strikeZ >= np.pi:
                strikeZ -= np.pi
            tempZ = np.array([z[0],z[1],0])
            if np.linalg.norm(tempZ) == 0:
                dipZ = np.pi / 2
            else:
                if np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z) > 1 or np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z) < -1:
                    if np.abs(np.abs(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)) - 1) < 0.00001:
                        dipZ = np.arccos(round(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)))
                    else:
                        raise Exception(np.abs(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z)))
                else:
                    dipZ = np.arccos(np.dot(tempZ, z) / np.linalg.norm(tempZ) / np.linalg.norm(z))
            if z[2] < 0:
                dipZ = -dipZ
                
            parameterX = np.append(parameterX, [strikeX,dipX])
            parameterZ = np.append(parameterZ, [strikeZ,dipZ])
            R = np.append(R, phi)
        parameterX = parameterX.reshape((-1,2))
        parameterZ = parameterZ.reshape((-1,2))
        return np.rad2deg(parameterX),np.rad2deg(parameterZ),R
    
class solution(picture):
    """Inversion driver: fitness, attractor and peak search.

    The driver ties the 4-D grid to the data: it evaluates the fitness of
    every grid point, searches the stress states that explain the observed
    fault slips and aggregates them into attractors.
    """

    def __init__(self,meshgrid = meshgrid(100,100,5).fourDimensionalGrid()):
        """Store the 4-D stress grid used by the inversion."""
        super().__init__()
        self.__meshgrid = meshgrid
        
    def setMeshgrid(self,meshgrid):
        """Replace the 4-D stress grid."""
        self.__meshgrid = meshgrid
        
    def getMeshgridLength(self):
        """Return the number of stress states in the grid."""
        return len(self.__meshgrid)
    
    def getMeshGrid(self):
        """Return the 4-D stress grid."""
        return self.__meshgrid
    
    def fitness(self,gridNumber,faultsParameter):
        """Misfit between the predicted and the observed slip of the faults.
        """
        self.setGrid(self.__meshgrid[gridNumber])
        self.setFault(faultsParameter)
        angleCos = np.dot(self.shearStressVectorTransform(),self.slipVector()) \
            / (np.linalg.norm(self.shearStressVectorTransform()) * np.linalg.norm(self.slipVector()))
        if np.abs(angleCos) > 1:
            if np.abs(angleCos - 1) < 0.001:
                angleCos = 1
            elif np.abs(1 + angleCos) < 0.001:
                angleCos = -1
            else:
                raise ValueError("angleCos is more than 1 or less than -1. Please check the data. shearStressVectorTransform is " 
                                 + str(self.shearStressVectorTransform()) + ", and slipVector is " + str(self.slipVector()) + 
                                 ". angleCos = " + str(angleCos))
        F = (1 - angleCos) / 2
        return F
    
    def randomTrialStress(self,amount):
        """Draw *amount* random stress states (alpha, beta, gamma, R)."""
        trialStress = np.zeros((amount,4))
        for index,row in enumerate(trialStress):
            trialStress[index,0] = np.random.uniform(0, 2 * np.pi)
            trialStress[index,1] = np.random.uniform(0, np.pi / 2)
            trialStress[index,2] = np.random.uniform(0, np.pi)
            trialStress[index,3] = np.random.uniform(0, 1)
        return trialStress
    
    def stressDifference(self,A,B):
        """Distance between two stress tensors, from their invariants."""
        D = A - B
        [I1,I2,I3] = self.invariantValues(D)
        realD = np.sqrt(2 * I1**2 - 4 * I2) / 3
        return realD
    
    def combinedVectorAverageLength(self,vectors):
        """Mean length of the summed vectors, normalised by the count."""
        c = np.sum(vectors,axis=0)
        return np.linalg.norm(c) / len(vectors)
    
    def attractor(self, tensors, oldAttractor, is5DVector = False):
        """Weighted mean stress state (attractor) of a set of tensors.

        Every tensor is weighted by the inverse of its distance to
        *oldAttractor*, so that near-by states dominate the result.
        """
        D = np.array([])
        tensorsWithWeight = np.array([])
        if is5DVector:
            for tensor in tensors:
                d = self.distance5D(tensor, oldAttractor)
                if d == 0:
                    d = 0.0000001
                D = np.append(D, d)
                tensorsWithWeight = np.append(tensorsWithWeight, tensor/d)
            tensorsWithWeight = np.reshape(tensorsWithWeight, (-1,5))
            tensorSum = np.sum(tensorsWithWeight, axis = 0)
            DSum = np.sum(1/D)
            newTesnsor = tensorSum / DSum
            return newTesnsor
        else:
            for tensor in tensors:
                d = self.stressDifference(tensor, oldAttractor)
                if d == 0:
                    d = 0.0000001
                D = np.append(D, d)
                tensorsWithWeight = np.append(tensorsWithWeight, tensor/d)
            tensorsWithWeight = np.reshape(tensorsWithWeight, (-1,3,3))
            tensorSum = np.sum(tensorsWithWeight, axis = 0)
            DSum = np.sum(1/D)
            newTesnsor = tensorSum / DSum
            return newTesnsor
    
    def calAttractor(self, vectorArray):
        """Iterate :meth:`attractor` until it converges, then resolve the axes.

        Returns the stress ratio Phi, the eigenvector matrix and the 5-D
        attractor itself.
        """
        self.setGrid(self.randomTrialStress(1)[0])
        oldAttractor = self.vector5D()
        #oldAttractor = vectorArray[0]
        newAttractor = self.attractor(vectorArray, oldAttractor, is5DVector = True)
        while self.distance5D(newAttractor, oldAttractor) > 0.0000001:
            oldAttractor = newAttractor
            newAttractor =self. attractor(vectorArray, oldAttractor, is5DVector = True)
        S = self.s5_to_stress_tensor(newAttractor,normalize=True)
        Phi, eigvecs = self.stressResolution(S, isArray=False)
        return Phi, eigvecs, newAttractor
    
    def stressResolution(self, tensor, isArray = True):
        """Eigen decomposition of a stress tensor -> Phi and eigenvectors.

        Eigenvalues are sorted in descending order (sigma1 >= sigma2 >= sigma3)
        and Phi = (s2 - s3) / (s1 - s3).  *isArray* selects vectorised input.
        """
        p = e = np.array([])
        if isArray:
            for i in tensor:
                eigvals, eigvecs = linalg.eigh(i)
                idx=np.argsort(eigvals)[::-1]
                eigvals=eigvals[idx]
                eigvecs=eigvecs[:,idx]
                s1,s2,s3=eigvals
                #print(eigvals)
                Phi=(s2-s3)/(s1-s3)
                p = np.append(p,Phi)
                e = np.append(e, eigvecs)
            e = np.reshape(e, (-1,3,3))
            return p,e
        else:
            eigvals, eigvecs = linalg.eigh(tensor)
            idx=np.argsort(eigvals)[::-1]
            eigvals=eigvals[idx]
            eigvecs=eigvecs[:,idx]
            s1,s2,s3=eigvals
            #print(eigvals)
            Phi=(s2-s3)/(s1-s3)
            return Phi, eigvecs
        
    def cal5DVectorDifference(self, vectors1, vectors2):
        """Angular difference of the principal axes of two sets of 5-D vectors."""
        theta = np.array([])
        Phi = np.array([])
        for (v1,v2) in zip(vectors1,vectors2):
            t1 = self.s5_to_stress_tensor(v1,normalize=True)
            t2 = self.s5_to_stress_tensor(v2,normalize=True)
            Phi1, eigvecs1 = self.stressResolution(t1,isArray=False)
            Phi2, eigvecs2 = self.stressResolution(t2,isArray=False)
            theta1 = np.arccos(np.abs(np.dot(eigvecs1[:,0], eigvecs2[:,0])))
            theta2 = np.arccos(np.abs(np.dot(eigvecs1[:,1], eigvecs2[:,1])))
            theta3 = np.arccos(np.abs(np.dot(eigvecs1[:,2], eigvecs2[:,2])))
            theta = np.append(theta, np.rad2deg([theta1,theta2,theta3]))
            Phi = np.append(Phi, [Phi1,Phi2])
        theta = np.reshape(theta, (-1,3))
        Phi = np.reshape(Phi, (-1,2))
        return theta, Phi
    
    def calDeviation(self, eigvecs, target = 'z'):
        """Deviation angle (degree) between the principal axes and a target axis.

        target = 'z'     : vertical axis
        target = 'north' : north direction
        """
        theta = np.array([])
        if target == 'z':
            z = np.array([0,0,1])
            for i in eigvecs:
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,0], z))))
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,1], z))))
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,2], z))))
            theta = np.reshape(theta, (-1,3))
        if target == 'north':
            north = np.array([0,-1,0])
            for i in eigvecs:
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,0], north))))
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,1], north))))
                theta = np.append(theta, np.arccos(np.abs(np.dot(i[:,2], north))))
            theta = np.reshape(theta, (-1,3))
            
        return np.rad2deg(theta)
    
    def fromEigvecsCalAzi(self, eigvecs):
        """Azimuth and stress regime derived from the principal axes.

        The principal axis closest to the vertical axis defines the regime:
        sigma1 near vertical -> Normal, sigma3 near vertical -> Reverse,
        otherwise -> Strike_Slip.  The azimuth is taken from the horizontal
        plane of the corresponding axis.
        """
        theta = self.calDeviation(eigvecs)
        azi = np.array([])
        aziType = []
        north = np.array([0,-1,0])
        for num,i in enumerate(eigvecs):
            index = np.where(theta[num] == np.min(theta[num]))[0]
            if index  == 0:
                temp = np.arccos(np.dot(north, [i[0,1], i[1,1], 0]))
                if temp < 0:
                    temp += np.pi
                azi = np.append(azi, temp)
                aziType.append('Normal')
            elif index == 1:
                temp = np.arccos(np.dot(north, [i[0,0], i[1,0], 0]))
                if temp < 0:
                    temp += np.pi
                azi = np.append(azi, temp)
                aziType.append('Strike_Slip')
            elif index == 2:
                temp = np.arccos(np.dot(north, [i[0,0], i[1,0], 0]))
                if temp < 0:
                    temp += np.pi
                azi = np.append(azi, temp)
                aziType.append('Reverse')
        return np.rad2deg(azi), aziType
        
    def peakFindInversion(self,faults,limitInDeg = 2,isMC = False):
        """Search the stress states that explain every observed fault.

        A grid point is kept when the misfit angle is below *limitInDeg*;
        with *isMC* the Mohr-Coulomb criterion must be fulfilled as well.
        """
        res = []
        for num,fault in enumerate(faults):
            ls = np.array([])
            for i in tqdm.trange(self.getMeshgridLength(),desc = 'fault ' + str(num) + '/' + str(len(faults) - 1) + ' with limit = ' + str(limitInDeg) + '°'):
                fit = self.fitness(i, fault)
                if isMC:
                    if fit < (1 - np.cos(np.deg2rad(limitInDeg))) / 2 and self.MC() > 0:
                        ls = np.append(ls, int(i))
                else:
                    if fit < (1 - np.cos(np.deg2rad(limitInDeg))) / 2:
                        ls = np.append(ls, int(i))
            stress = self.getMeshGrid()[ls.astype(int)]
            res.append(stress.reshape((-1,4)))
        return res
    
    def faultsCreateSimulation(self,faultsNumberForPerStress,stress,noise,isGausssian = True,isDeg = True,isSeed = False,seed = 1):
        """Generate synthetic faults that are consistent with given stresses.

        faultsNumberForPerStress : number of faults per stress state
        stress   : stress states (degrees when *isDeg*, radians otherwise)
        noise    : noise amplitude added to the slip angle, in degrees or radians
        isGausssian : Gaussian noise (True) or uniform noise (False)
        isSeed   : fix the random seed (True) or not (False)
        """
        faultsNumber = faultsNumberForPerStress * len(stress)
        faults = np.zeros((faultsNumber,3))
        count = 0
        for i in stress:
            if isDeg:
                self.setGridInDeg(i)
            else:
                self.setGrid(i)
            for j in range(faultsNumberForPerStress):
                if isSeed:
                    np.random.seed(seed)
                    faults[count,0] = np.random.uniform(low = 0, high = np.pi * 2)
                    np.random.seed(seed)
                    faults[count,1] = np.random.uniform(low = 0, high = np.pi / 2) 
                else:
                    faults[count,0] = np.random.uniform(low = 0, high = np.pi * 2)
                    faults[count,1] = np.random.uniform(low = 0, high = np.pi / 2)
                self.setFault(faults[count])
                cos = np.dot(self.shearStressVectorTransform(), self.strikeVector())
                
                if np.abs(cos) > 1:
                    cos = cos / np.linalg.norm(self.shearStressVectorTransform()) / np.linalg.norm(self.strikeVector())
                    if np.abs(cos) < 1.01 and np.abs(cos) > 1:
                        cos = round(cos)
                if np.dot(np.cross(self.shearStressVectorTransform(), self.strikeVector()), self.faultNormalVector()) > 0:
                    slip = np.arccos(cos)
                else:
                    slip = -np.arccos(cos)
                if isDeg:
                    if isGausssian:
                        randomNoise = np.random.normal(0,np.deg2rad(noise))
                    else:
                        randomNoise = np.random.uniform(low = -np.deg2rad(noise), high = np.deg2rad(noise))
                else:
                    if isGausssian:
                        randomNoise = np.random.normal(0,noise)
                    else:
                        randomNoise = np.random.uniform(low = -noise, high = noise)
                faults[count,2] = slip + randomNoise
                count += 1
                  
        return faults
    
    def customFaultsSlipCalculation(self, faults, stress):
        """Compute the slip angle of faults for a given stress state.

        faults is an n x 2 array whose columns are the strike and the dip of
        every fault; the returned array adds the corresponding slip angle.
        """
        self.setGrid(stress)
        newFaults = np.array([])
        for (strike, dip) in faults:
            self.setFault([strike, dip, 0])
            cos = np.dot(self.shearStressVectorTransform(), self.strikeVector())
            
            if np.abs(cos) > 1:
                cos = cos / np.linalg.norm(self.shearStressVectorTransform()) / np.linalg.norm(self.strikeVector())
                if np.abs(cos) < 1.01 and np.abs(cos) > 1:
                    cos = round(cos)
            if np.dot(np.cross(self.shearStressVectorTransform(), self.strikeVector()), self.faultNormalVector()) > 0:
                slip = np.arccos(cos)
                #print("slip = " + str(np.rad2deg(slip)))
            else:
                slip = -np.arccos(cos)
            newFaults = np.append(newFaults, [strike, dip, slip])
        newFaults = np.reshape(newFaults, (-1, 3))
        return newFaults
 
