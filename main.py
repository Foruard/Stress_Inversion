# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:06:26 2026

@author: 77891

Example driver: inverts the focal mechanisms of the Bohai area, selects for
every earthquake the nodal plane that best agrees with the World Stress Map
and draws the stress maps together with the fault activation risk map.
"""

import singleFaultInversion as sfi
import pandas as pd
import numpy as np


# Create the inversion driver. The default 4-D grid is 100 directions x
# 100 rotations x 5 stress ratios, which is enough for the examples below.
run = sfi.solution()
# A custom grid (e.g. one that was computed in advance) can be used instead:
#run.setMeshgrid(pd.read_csv('./data/mesh_360_180_10.csv',index_col=0).values)

# ---------------------------------------------------------------------------
# Optional synthetic test (kept disabled).
# Two faults are generated from manually chosen stress tensors, inverted and
# compared, which shows how the inversion reacts to a different stress state.
# ---------------------------------------------------------------------------

fault1 = run.customFaultsSlipCalculation(np.deg2rad([[0,30]]), [0,np.pi/2,0,0.5])
fault2 = run.customFaultsSlipCalculation(np.deg2rad([[0,30]]), [0,0,0,0.5])
stress1 = run.peakFindInversion(fault1)
stress2 = run.peakFindInversion(fault2)


run.setColor('coolwarm')
run.setDpi(600)
Fig4a = run.visualization(stress1,isShowTitle=True)
Fig4b = run.visualization(stress2,isShowTitle=True)



# ---------------------------------------------------------------------------
# 1. Read the focal mechanism catalogue of the Bohai area.
#    Columns 0-2 : strike, dip, slip of the primary nodal plane
#    Columns 3-5 : strike, dip, slip of the conjugate nodal plane
#    Columns 6-7 : latitude, longitude
# ---------------------------------------------------------------------------
BohaiFocalData = pd.read_csv('./data/focal_data.csv',index_col=1).values[:, 1:10]
faults = np.deg2rad(BohaiFocalData[:,:3])
cfaults = np.deg2rad(BohaiFocalData[:,3:6])
BohaiFaultsLocation = BohaiFocalData[:,[6,7]]

# Keep the first two events only (quick demonstration of the workflow).
faults = faults[:2]
cfaults = cfaults[:2]
BohaiFaultsLocation = BohaiFaultsLocation[:2]

# ---------------------------------------------------------------------------
# 2. Invert both nodal planes independently. limitInDeg is the tolerated
#    misfit angle (in degrees) between predicted and observed slip.
# ---------------------------------------------------------------------------
stresses = run.peakFindInversion(faults, limitInDeg = 2)
cstresses = run.peakFindInversion(cfaults, limitInDeg = 2)

# ---------------------------------------------------------------------------
# 3. Primary nodal plane: convert every accepted stress state into a 5-D
#    vector, search its attractor and derive Phi and the principal axes.
# ---------------------------------------------------------------------------
BohaiEigvecs = np.array([])
BohaiPhi = np.array([])
for stress in stresses:
    vector = run.vector5D(stress)
    # Attractor = mean stress state of all accepted grid points
    Phi, eigvecs, attractor = run.calAttractor(vector)
    BohaiEigvecs = np.append(BohaiEigvecs, eigvecs)
    BohaiPhi = np.append(BohaiPhi, Phi)
BohaiEigvecs = np.reshape(BohaiEigvecs, (-1,3,3))
# Azimuth and stress regime of the primary nodal plane solutions.
azi,aziType = run.fromEigvecsCalAzi(BohaiEigvecs)


# ---------------------------------------------------------------------------
# 4. Conjugate nodal plane: exactly the same treatment.
# ---------------------------------------------------------------------------
cBohaiEigvecs = np.array([])
cBohaiPhi = np.array([])
for cstress in cstresses:
    vector = run.vector5D(cstress)
    Phi, eigvecs, attractor = run.calAttractor(vector)
    cBohaiEigvecs = np.append(cBohaiEigvecs, eigvecs)
    cBohaiPhi = np.append(cBohaiPhi, Phi)
cBohaiEigvecs = np.reshape(cBohaiEigvecs, (-1,3,3))
# Azimuth and stress regime of the conjugate nodal plane solutions.
cazi,caziType = run.fromEigvecsCalAzi(cBohaiEigvecs)

# World Stress Map database used later for the selection step.
wsm = './data/WSM_Database_2025.csv'
# Length of the stress direction segments on the maps (degrees).
run.setLineLength(2)
# Plate boundary points drawn on top of the maps.
boundary = run.getBoundary()


# ---------------------------------------------------------------------------
# 5. Draw the two candidate stress maps (primary and conjugate plane).
# ---------------------------------------------------------------------------
Fig8 = run.stressMapPlus(lon = BohaiFaultsLocation[:,1], lat = BohaiFaultsLocation[:,0], azi = azi, 
                         aziType = aziType, background = './background/Satellite_map.png', 
                         area = [111,130,30,45], isTitle = False, boundaryPoints = boundary, 
                         boundaryColor = 'yellow', boundarySize = 10, boundaryLine = False, 
                         boundaryLineWidth = 0.8)
Fig9 = run.stressMapPlus(lon = BohaiFaultsLocation[:,1], lat = BohaiFaultsLocation[:,0], azi = cazi, 
                         aziType = caziType, background = './background/Satellite_map.png', 
                         area = [111,130,30,45], isTitle = False, boundaryPoints = boundary, 
                         boundaryColor = 'yellow', boundarySize = 10, boundaryLine = False, 
                         boundaryLineWidth = 0.8)
    


# ---------------------------------------------------------------------------
# 6. Compare both candidates with the nearest World Stress Map station.
#    The rows are [azi1, aziType1, azi2, aziType2, lat, lon].
# ---------------------------------------------------------------------------
primary = np.append(azi, aziType).reshape((-1,len(faults))).T
conjugate = np.append(cazi, caziType).reshape((-1,len(faults))).T
data = np.append(primary, conjugate, axis=1)
data = np.append(data, BohaiFaultsLocation, axis=1)

# res[:, 2] is 0 when the primary plane was kept and 1 for the conjugate one.
res = run.matchAziWithDatabase(wsm, data)
finalEigvecs = np.array([])
finalPhi = np.array([])
finalFocal = np.array([])
for num,i in enumerate(res[:,2]):
    if i == 0:
        # Primary nodal plane wins: keep its axes, Phi and focal parameters
        finalEigvecs = np.append(finalEigvecs, BohaiEigvecs[num])
        finalPhi = np.append(finalPhi, BohaiPhi[num])
        finalFocal = np.append(finalFocal, BohaiFocalData[num, [0,1,2,6,7]])
    else:
        # Conjugate nodal plane wins: keep its axes, Phi and focal parameters
        finalEigvecs = np.append(finalEigvecs, cBohaiEigvecs[num])
        finalPhi = np.append(finalPhi, cBohaiPhi[num])
        finalFocal = np.append(finalFocal, BohaiFocalData[num, [3,4,5,6,7]])
finalEigvecs = np.reshape(finalEigvecs, (-1,3,3))
finalFocal = np.reshape(finalFocal, (-1,5))


# ---------------------------------------------------------------------------
# 7. Selected stress map, the same map with the database stations and the
#    fault activation risk map.
# ---------------------------------------------------------------------------
Fig10 = run.stressMapPlus(lon = BohaiFaultsLocation[:,1], lat = BohaiFaultsLocation[:,0], 
                          azi = res[:,0], aziType = res[:,1], background = './background/Satellite_map.png', 
                          area = [111,130,30,45], isTitle = False, boundaryPoints = boundary,
                          boundaryColor = 'yellow', boundarySize = 10, boundaryLine = False)

Fig11 = run.stressMapWithDatabase(lon = BohaiFaultsLocation[:,1], lat = BohaiFaultsLocation[:,0],
                                  azi = res[:,0], aziType = res[:,1], background = './background/Satellite_map.png',
                                  area = [111,130,30,45], csvFile = wsm, boundaryPoints=boundary, isTitle = False,
                                  numTicks = 5, boundaryColor='yellow', boundarySize=15, boundaryLine=False,
                                  boundaryLineWidth=0.8)

# risk holds the per-earthquake activation risk; the map itself is drawn
# from an RBF interpolation of those values.
Fig12, risk = run.FaultsActivationRiskMap(eigvecs = finalEigvecs, Phi = finalPhi, focalData = finalFocal, 
                                          background = './background/Shadow_map.png', 
                                          area = [111,130,30,45], isTitle = False, smoothing = 0.5,
                                          numTicks = 5, colorbarPad = 0.15, terrainStrength=0.45,
                                          saturationScale = 1.2, boundaryPoints = boundary,
                                          boundaryColor = 'yellow', boundarySize = 15, boundaryLine = False,
                                          boundaryLineWidth = 0.8)

