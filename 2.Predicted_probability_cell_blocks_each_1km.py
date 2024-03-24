import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Load the Hotspots data
df = pd.read_csv('C:/Users/amilaw/Desktop/Ignition Modelling project/Data/Data summary 3_Lightning/ZZ_Model_development/ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category/Random Forest/Predicted probability maps/1_predicted_probability.csv')

# Define the grid parameters
grid_resolution = 0.03
min_latitude = -44.0
max_latitude = -39.0
min_longitude = 143.0
max_longitude = 149.0

# Initialize lists to store the grid cell boundaries
lower_latitudes = []
higher_latitudes = []
lower_longitudes = []
higher_longitudes = []

# Iterate through each row in the DataFrame
for _, row in df.iterrows():
    latitude = row['Lat']
    longitude = row['Lon']

    # Calculate grid cell boundaries for the given coordinates
    lower_lat = np.floor((latitude - min_latitude) / grid_resolution) * grid_resolution + min_latitude
    higher_lat = lower_lat + grid_resolution
    lower_lon = np.floor((longitude - min_longitude) / grid_resolution) * grid_resolution + min_longitude
    higher_lon = lower_lon + grid_resolution

    # Append the boundaries to the lists
    lower_latitudes.append(lower_lat)
    higher_latitudes.append(higher_lat)
    lower_longitudes.append(lower_lon)
    higher_longitudes.append(higher_lon)

# Add the boundary columns to the DataFrame
df['Lower_Latitude'] = lower_latitudes
df['Higher_Latitude'] = higher_latitudes
df['Lower_Longitude'] = lower_longitudes
df['Higher_Longitude'] = higher_longitudes

# Save the updated DataFrame to a new CSV file
output_file = 'C:/Users/amilaw/Desktop/Ignition Modelling project/Data\Data summary 3_Lightning/ZZ_Model_development/ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category/Random Forest/Predicted probability maps/2_Predicted_probability.CSV'
df.to_csv(output_file, index=False)

