---
layout: default
title: Download data sets
description: Links to download data sets used.
---

## Links to download data sets:
- [Processed images and spectroscopic redshifts from Pasquet et al. 2019](https://deepdip.iap.fr/#item/60ef1e05be2b8ebb048d951d)
- [Additional catalogs and redshift predictions from the paper](www.d-scholarship.pitt.edu)

**A breif description of the data files provided:**
- `sdss.npz`: NumPy `.npz` file containing both the image datacube and the labels used for training and testing the photo-z classifier of [Pasquet et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...621A..26P). The datacube ("cube") contains images with size 64×64 pixels for 659857 SDSS galaxies in the u,g,r,i and z filters as a NumPy float32 ndarray with shape (659857, 64, 64, 5). The labels array ("labels") is a NumPy recarray with 659857 lines (one for each galaxy) of 64 columns. More details can be found at the [data website](https://deepdip.iap.fr/#item/60ef1e05be2b8ebb048d951d) or [Pasquet et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...621A..26P).
- 

If you have any questions, comments or want to collaborate, feel free to drop an email to: biprateep@pitt.edu

[Back to homepage](./)
