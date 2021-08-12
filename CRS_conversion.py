# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:07:31 2021

@author: HamishMitchell
"""
# Imports
import os
import pandas as pd
import geopandas as gpd
from osgeo import gdal

os.chdir("C:/Users/HamishMitchell/gis_uk/RasterData/RasterData")

# Raster Tiff to a XYZ (csv)
def translateDEM_CSV(raster, outXYZ):
    ds = gdal.Open(raster)
    xyz = gdal.Translate(outXYZ, ds)
    xyz = None

# Read CSV
def readXYZ(outXYZ, col):
    df = pd.read_csv(outXYZ, sep = " ", header=None)
    print(col)
    print(f"Dataframe has the shape: {df.shape}")
    df.columns = ['x', 'y', col]
    print('-'*40)
    return df

# Set x, y index
def set_index(var):
    var.set_index(['x','y'], inplace=True)
    return var

# Convert to shapefile for CRS change
def csv_2_shp(df, x, y, out_shp):
    # Shapefile Conversion
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df[x], df[y]))
    gdf = gdf.set_crs(epsg=32630)
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(out_shp)
    return gdf

# Convert to CSV for outputs
def shp_2_csv(gdf, z, out_csv):
    newDf = pd.DataFrame()
    newDf['x'] = gdf.geometry.x
    newDf['y'] = gdf.geometry.y
    newDf[z] = gdf[z]
    newDf.to_csv(out_csv)
    return newDf


## FUNCTION CALLS
# Test 1
land = pd.read_csv('scripts/LandslideProbabilities.csv')
csv_2_shp(land, 'x', 'y','crs_test.shp')
gdf = gpd.read_file('crs_test.shp')
shp_2_csv(gdf, 'Landslide_', 'testConvertedCRS.csv')


# LANDSLIDE CONVERSION
#translateDEM_CSV("LandslideHyperFilled.tif", 'xyz/LandslideProbabilities.xyz') # only do once
land = readXYZ('xyz/LandslideProbabilities.xyz', 'LSprob')
DEM = readXYZ("xyz/DEM.xyz", "z")
# Round XY before concat
DEM = DEM.round(0)
land[['x','y']] = land[['x', 'y']].round(0)
# Set index
DEM = set_index(DEM)
land = set_index(land)
df = pd.concat([DEM, land], axis=1, join='outer')

# Crop df by the DEM 
cropDf = df[df['z'] > -5]
cropDf.shape
cropDf = cropDf.drop('z', axis=1) # get rid of DEM so just LS output


cropDf = cropDf.reset_index()
csv_2_shp(cropDf, 'x', 'y','LSProbs.shp')
ls = gpd.read_file('LSProbs.shp')
shp_2_csv(ls, 'LSprob', 'LSprob_wgs1984.csv')




# DISPLACEMENT CONVERSION 
translateDEM_CSV("RFDisplacIW123.tif", 'xyz/RFDispAll.xyz') # only do once
disp = readXYZ('xyz/RFDispAll.xyz', 'Disp_m_yr')
DEM = readXYZ("xyz/DEM.xyz", "z")
# Round XY before concat
DEM = DEM.round(0)
disp[['x','y']] = disp[['x', 'y']].round(0)
# Set index
DEM = set_index(DEM)
disp = set_index(disp)
df = pd.concat([DEM, disp], axis=1, join='outer')

# Crop df by the DEM 
cropDf = df[df['z'] > -5]
cropDf.shape
cropDf = cropDf.drop('z', axis=1) # get rid of DEM so just disp output

cropDf = cropDf.reset_index()
csv_2_shp(cropDf, 'x', 'y','RFDisplacIW123.shp')
ls = gpd.read_file('RFDisplacIW123.shp')
shp_2_csv(ls, 'Disp_m_yr', 'RFDisplacIW123_wgs1984.csv')






### FOR QGIS TEST ONLY (IGNORE)
df = pd.read_csv("scripts/iw2TunedSubsidence.csv")
# Shapefile Conversion
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['x'], df['y']))
gdf = gdf.set_crs(epsg=32630)

gdf.to_file("iw2SubsUTM.shp")