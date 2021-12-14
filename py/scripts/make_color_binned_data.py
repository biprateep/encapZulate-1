"""
Bin the data into equal population bins based on rest frame color
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path("/bgfs/jnewman/bid13/photoZ/data/pasquet2019/") #CRC
# data_path = Path("/data/bid13/photoZ/data/pasquet2019/")  # Dara

num_class = 10
keep_frac = 0.5


data=np.load(data_path/"sdss_vagc.npz", allow_pickle=True)

cat=data["labels"] #Sdss data


cat = pd.DataFrame(cat)

#clean up
cat = cat.replace([np.inf, -np.inf], np.nan)
cat = cat.dropna()

#Pasquet cuts
cat = cat[(cat['dered_petro_r']<=17.8) & (cat['z']<=0.4) & (cat['z']>=0)]
imageID = np.array(cat["imageID"])

#define the colors
col_gr = cat["absMag_g"]-cat["absMag_r"]

#divide in bins
labels,bins = pd.qcut(col_gr,num_class,labels=False, retbins=True, precision=5)



cube = (data["cube"][imageID]).astype("float16")
labels = np.array(labels)
specObjID = data["labels"]["specObjID"][imageID]
z = data["labels"]["z"][imageID]
cat = data["labels"][imageID]


if keep_frac <1 :
    np.random.seed(200)
    mask = np.random.choice([True, False], size=len(z), p = [keep_frac, 1-keep_frac])
    
    cube_extra = cube[~mask]
    cube = cube[mask]
    
    labels_extra = labels[~mask]
    labels = labels[mask]
    
    z_extra = z[~mask]
    z = z[mask]
    
    specObjID_extra = specObjID[~mask]
    specObjID = specObjID[mask]
    
    cat_extra = cat[~mask]
    cat = cat[mask]
    
    np.savez(
        data_path/ ("sdss_gr_"+str(num_class)+"_"+str(keep_frac)+".npz"),
        cube=cube,
        labels=labels,
        specObjID=specObjID,
        z=z,
        cat=cat,
        cube_extra = cube_extra,
        labels_extra = labels_extra,
        specObjID_extra = specObjID_extra,
        cat_extra = cat_extra,   
        )
else:
    np.savez(
        data_path/ ("sdss_gr_"+str(num_class)+".npz"),
        cube=cube,
        labels=labels,
        specObjID=specObjID,
        z=z,
        cat=cat,
        )
