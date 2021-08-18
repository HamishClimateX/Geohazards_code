# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:32:43 2021

@author: HamishMitchell
"""
# IMPORT PACKAGES
import os 
import glob 
import subprocess

import pandas as pd
from osgeo import gdal
from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Check the pwd is correct    
os.chdir("C:/Users/HamishMitchell/gis_uk/RasterData/RasterData")


# FEATURES TO CSV
# Raster Tiff to a XYZ
def translateDEM_CSV(raster, outXYZ):
    ds = gdal.Open(raster)
    xyz = gdal.Translate(outXYZ, ds)
    xyz = None

def readXYZ(outXYZ, col):
    df = pd.read_csv(outXYZ, sep = " ", header=None)
    print(col)
    print(f"Dataframe has the shape: {df.shape}")
    df.columns = ['x', 'y', col]
    print('-'*40)
    return df

# Mask, check the number of NaN values
def DFmask(var, col, minVal):
    # Its essential for the model that all the inputs have the same df shape
    print(f"Dataframe has {var[var[col] < -10].sum()} NaN values")
    var = var[var[col]> minVal]
    print(f"Dataframe after mask has the shape: {var.shape}")
    print('-'*40)
    return var

# Mask, check the number of NaN values
def DFmaskVeg(var, col, minVal):
    # Its essential for the model that all the inputs have the same df shape
    print(f"Dataframe has {var[var[col] < -10].sum()} NaN values")
    var = var[var[col]>= minVal]
    print(f"Dataframe after mask has the shape: {var.shape}")
    print('-'*40)
    return var
    
# Set x, y index
def set_index(var):
    var.set_index(['x','y'], inplace=True)
    return var
    


# Define the input files
inputs = ["Thames100m.tif", "ThamesSlope.tif", "ThamesAspect.tif",
          "Bedrock.tif", "ShrSwell.tif", "SuperficialRaster.tif", 
          "LandUseAligned.tif", "VegetationClip100m.tif", "DFAlign.tif", "DistanceRivers.tif",
          "Roads.tif"]
outputs = ["xyz/DEM.xyz", "xyz/Slope.xyz", "xyz/Aspect.xyz",
           "xyz/Bedrock.xyz", "xyz/ShSwell.xyz", "xyz/SuperfDep.xyz", 
           "xyz/Land.xyz", "xyz/Veg.xyz", "xyz/DistFaults.xyz", "xyz/DistRiv.xyz",
           "xyz/Roads.xyz"]

# Run the translate function for all rasters; use glob if appropriate (i.e., no non-df tiffs in the directory)
for inp, out in zip(inputs, outputs):
    translateDEM_CSV(inp, out)


translateDEM_CSV('GrdMotion.tif', 'xyz/DisplacementRate.xyz')
translateDEM_CSV('DistArtificGround.tif', 'xyz/ArtificialGrnd.xyz')


# Define feature variables
DEM = readXYZ("xyz/DEM.xyz", "z")
slope = readXYZ("xyz/Slope.xyz", "s")
aspect = readXYZ("xyz/Aspect.xyz", "asp")
rock = readXYZ("xyz/Bedrock.xyz", "RockClass")
sswell = readXYZ("xyz/ShSwell.xyz", "ShSwell")
superf = readXYZ("xyz/SuperfDep.xyz", "SupfClass")
riv = readXYZ("xyz/DistRiv.xyz", "RiverDistance_m")
roads = readXYZ("xyz/Roads.xyz", "RoadDistance_m")
# Below have a different grid
land = readXYZ("xyz/Land.xyz", "LUC") # 2923340
veg = readXYZ("xyz/Veg.xyz", "NVDI") # 2923340
faults = readXYZ("xyz/DistFaults.xyz", "FaultDistance_m") # 2918484
# Target classes
landslides = readXYZ('xyz/landslide.xyz', 'Landslide')
sub = readXYZ('iw2DisplacementRate.xyz', "GroundMotion_m_yr")
artificial = readXYZ('xyz/ArtificialGrnd.xyz', 'ArtificialDistance_m')


variables = [DEM, slope, aspect, rock, sswell, superf, riv, roads, artificial,
             land, veg, faults, sub]
columns = ['z', 's', 'RockClass', 'ShSwell', 'SupfClass', 'RiverDistance_m', 
           'RoadDistance_m', 'ArtificialDistance_m', 'LUC', 'NVDI', 'FaultDistance_m', 'GroundMotion_m_yr'] # landslide and aspect should not be here


"""
# Doesn't seem to work, dont know why, I'm too tired to figure out
for var, col in zip(variables, columns):
    var = var.round()
    print(var.head())
    print('-'*40)
"""




sub["GroundMotion_m_yr"][sub["GroundMotion_m_yr"] < -100] = None
sub = sub.dropna()


# Round XY
DEM = DEM.round()
slope = slope.round()
aspect = aspect.round()
rock = rock.round()
sswell = sswell.round()
superf = superf.round()
riv = riv.round()
roads = roads.round()
# Below have a different grid
land = land.round()
veg = veg.round()
faults = faults.round()
landslides = landslides.round()
artificial = artificial.round()
sub[['x','y']] = sub[['x', 'y']].round(0)
sub.head()

# Set the index
DEM = set_index(DEM)
slope = set_index(slope)
aspect = set_index(aspect)
rock = set_index(rock)
sswell = set_index(sswell)
superf = set_index(superf)
riv = set_index(riv)
roads = set_index(roads)
artificial = set_index(artificial)
land = set_index(land)
veg = set_index(veg)
faults = set_index(faults)
# Target classes
landslides = set_index(landslides)
sub = set_index(sub)


# Grid 1 concat
g1 = [DEM, slope, aspect, rock, sswell, superf, 
      riv, roads, artificial, landslides, sub]
# Concatenate the df 
df = pd.concat(g1, axis=1, join='outer')

# Grid 2 concat
g2 = [land, veg, faults]
# Concatenate df2
df2 = pd.concat(g2, axis=1, join='outer')


# Final concat
dframe = pd.concat([df, df2], axis=1, join='outer')
dframe.shape
dframe.to_csv('iw2_subs.csv')


# Mask dataframe by the DEM 
DF = dframe[dframe['z'] > -5]
DF.shape
DF.isna().sum()
DF['GroundMotion_m_yr'].sample(50)


# Replace -9999 with None for conditioning factors 
for col in columns:
    DF[col][DF[col] == -9999] = None
DF['GroundMotion_m_yr'][DF['GroundMotion_m_yr'] < -1000] = None

# Replace -9999 with 0 in landslide class
DF['Landslide'][DF['Landslide'] == -9999] = 0
DF['asp'][DF['asp'] == -9999] = 0

# Check the proportion of positive instances in the landslide class
proportion_landslide = len(DF[DF['Landslide']==1])/len(DF) 


# Save landslide dataframe as CSV 
DF.shape
DF.to_csv('iw2df.csv')
