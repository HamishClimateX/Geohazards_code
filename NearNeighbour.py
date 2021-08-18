# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:04:42 2021

@author: HamishMitchell
"""
# IMPORT PACKAGES
import io
import os 
import glob 
import subprocess
from zipfile import ZipFile


from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Check the pwd is correct    
os.getcwd()
os.chdir("C:/Users/HamishMitchell/gis_uk/RasterData/RasterData")


# Read shapefile zip folder
def read_gdf_from_zip(zip_fp):
    """
    Reads a spatial dataset from ZipFile into GeoPandas. Assumes that there is only a single file (such as GeoPackage) 
    inside the ZipFile.
    """
    with ZipFile(zip_fp) as z:
        # Lists all files inside the ZipFile, here assumes that there is only a single file inside
        layer = z.namelist()[0]
        data = gpd.read_file(z.read(layer))
    return data

def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=True):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    
    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format 
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    
    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    
    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)
    
    # Add distance if requested 
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius
        
    return closest_points



# CALL FUNCTIONS
# Define Input Filepaths
roads = gpd.read_file('NearestNeigbour/RoadsPoints.shp')
points = gpd.read_file('NearestNeigbour/ThamesSinglePoints.shp')

# Check the crs
roads.crs
points.crs
roads = roads.set_crs(epsg=32630)

# Change the CRS
roads = roads.to_crs(epsg=4326)
points = points.to_crs(epsg=4326)


# Call the nearest neighbour function 
nearestRoad = nearest_neighbor(points, roads)

# Now we should have exactly the same number of nearest roads as we have points
print(len(nearestRoad), '==', len(points))

# Check the output file; set the crs
points['distance'] = nearestRoad['distance']
points.crs
points = points.set_crs(epsg=4326)

points.head()
points.shape
points.isna().sum()

# Output to a shapefile
points.to_file("RoadSimplified.shp")
