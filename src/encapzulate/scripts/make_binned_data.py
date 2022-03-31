"""
Create binned data with quality cuts from Pasquet 2019 Data.
NOTE: Needs to run on a high memory node (Should have atleast 100Gb of RAM)
author: biprateep
"""

import numpy as np

dataPath="/bgfs/jnewman/bid13/photoZ/data/pasquet2019/" #Path on CRC
#dataPath="/data/bid13/photoZ/data/pasquet2019/" #Path on Dara

data=np.load(dataPath+"sdss.npz")

cube=data["cube"] #Images
labels=data["labels"] #Sdss data

z_min = 0.1
dz = 2.2e-3
num_class = 22
keep_frac = 0.01

# Dynamically calculate z_max
z_max = z_min + num_class*dz

# apply quality cuts
cuts = ( (labels['dered_petro_r']<17.8) & (labels['z']<=z_max) & (labels['z']>=z_min) )

cube = cube[cuts]
labels = labels[cuts]
z = labels["z"]
specObjID = labels["specObjID"]

bin_edges=np.arange(z_min,z_max+dz,dz)

labels=np.digitize(labels["z"],bin_edges)-1

if keep_frac <1 :
    mask = np.random.choice([True, False], size=len(z), p = [keep_frac, 1-keep_frac])
    cube = cube[mask]
    labels = labels[mask]
    z = z[mask]
    specObjID = specObjID[mask]
    
    np.savez(dataPath + "sdss_unittest_z_"+str(z_min)+"-"+str(z_max)+"_bins_"+str(num_class)+"_frac_"+str(keep_frac)+".npz", cube=cube, labels=labels, z=z, specObjID=specObjID)

else:
    np.savez(dataPath + "sdss_binned_z_"+str(z_min)+"-"+str(z_max)+"_bins_"+str(num_class)+".npz", cube=cube, labels=labels, z=z, specObjID=specObjID)
