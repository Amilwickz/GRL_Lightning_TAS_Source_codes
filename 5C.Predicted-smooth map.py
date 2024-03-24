import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np

# Read the CSV file
csv_file = r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\ZZ_Model_development\ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category\Random Forest\Predicted probability maps\3_Predicted_probability_cell_each_1km.CSV'
df = pd.read_csv(csv_file)

# Read the shapefile
shapefile = r'C:\Users\amilaw\Desktop\Ignition Modelling project\Lightning_FMC_Project\2.Data\Shapefiles\Creating Tas plain shapefile\Tasmania_plain.shp'
gdf_tasmania = gpd.read_file(shapefile)
gdf_tasmania = gdf_tasmania.to_crs('EPSG:4326')

# Define the extent of the grid
minimum_latitude = -44.0
maximum_latitude = -39.0
minimum_longitude = 143.0
maximum_longitude = 149.0

# Create a new GeoDataFrame for the grid cells
geometry = [Polygon([(row['Lower_Longitude'], row['Lower_Latitude']),
                    (row['Higher_Longitude'], row['Lower_Latitude']),
                    (row['Higher_Longitude'], row['Higher_Latitude']),
                    (row['Lower_Longitude'], row['Higher_Latitude'])])
           for idx, row in df.iterrows()]
grid_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Merge grid data with Tasmania shapefile to get common geometries
merged_gdf = gpd.sjoin(grid_gdf, gdf_tasmania, how='inner', op='intersects')

# Create a regular grid to interpolate values
xi, yi = np.meshgrid(np.linspace(minimum_longitude, maximum_longitude, 167),
                    np.linspace(minimum_latitude, maximum_latitude, 167))

# Interpolate the log of the mean predicted probabilities
zi = griddata((merged_gdf['Lower_Longitude'], merged_gdf['Lower_Latitude']),
             np.log(merged_gdf['Mean Predicted Probability']),
             (xi, yi), method='cubic')

# Apply Gaussian smoothing to the interpolated data
zi_smooth = gaussian_filter(zi, sigma=0.1)  # Adjust sigma for the desired level of smoothing

# Create a mask using the shapefile boundary
mask = gdf_tasmania.geometry.unary_union

# Set values outside the mask to np.nan
zi_smooth[~np.array([point.within(mask) for point in gpd.points_from_xy(xi.flatten(), yi.flatten())]).reshape(xi.shape)] = np.nan

# Plot the smoothed map with values outside the mask in white
fig, ax = plt.subplots(figsize=(6, 6))
c = ax.contourf(xi, yi, zi_smooth, cmap='viridis', extend='both', levels=4000)  # Use a large number of levels for smooth gradient
gdf_tasmania.boundary.plot(ax=ax, linewidth=0.5)  # Plot the boundary
gdf_tasmania.plot(ax=ax, facecolor='none')  # Plot the shapefile

# Add a continuous vertical colorbar with only 'Low' and 'High' labels
cbar = plt.colorbar(c, ax=ax, orientation='vertical')
cbar.set_ticks([])  # Remove all ticks

# Add 'Low' and 'High' labels at the two ends of the colorbar
cbar.ax.text(3, 0.0, 'Low', ha='center', va='center', fontsize=10)
cbar.ax.text(3, 0.95, 'High', ha='left', va='center', fontsize=10)

# Show the plot
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(minimum_longitude, maximum_longitude)
plt.ylim(minimum_latitude, maximum_latitude)

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.show()

