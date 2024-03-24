#@Amil-UTAS: This code use the predicted probability values and convert those 'cell-belonged' values into a smooth regions by 'cubic' interpolation method.
#This helps to create regions of vulnerable regions (lightning-caused wildfires)
#To vary the resolution of the regions/inter[olation, I can change the values given in the '# Create a regular grid to interpolate values' section. As I have the max, min, lat and lon values, I can get the resolution I want by adjusting this value like 100, 200, 500, 100 etc.


import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches  # Importing mpatches for legend rectangles


# Define color thresholds
thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = ['yellow', 'red', 'limegreen', 'navy', 'black']

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

# Interpolate the mean predicted probabilities
zi = griddata((merged_gdf['Lower_Longitude'], merged_gdf['Lower_Latitude']),
              merged_gdf['Mean Predicted Probability'],
              (xi, yi), method='cubic')

# Create a mask using the shapefile boundary
mask = gdf_tasmania.geometry.unary_union

# Set values outside the mask to np.nan
zi[~np.array([point.within(mask) for point in gpd.points_from_xy(xi.flatten(), yi.flatten())]).reshape(xi.shape)] = np.nan

# Plot the interpolated map with values outside the mask in white
fig, ax = plt.subplots(figsize=(10, 10))
c = ax.contourf(xi, yi, zi, levels=thresholds, colors=colors, alpha=0.7, extend='both')
gdf_tasmania.boundary.plot(ax=ax, linewidth=0.5)  # Plot the boundary
gdf_tasmania.plot(ax=ax, facecolor='none')  # Plot the shapefile

# Create a legend with colored rectangles for mean predicted probability values
legend_handles = [
    mpatches.Patch(color=color, label=f'{thresholds[i]:.1f}-{thresholds[i + 1]:.1f}')
    for i, color in enumerate(colors)
]
plt.legend(handles=legend_handles, loc='center left', framealpha=0.5, frameon=True, title='Mean Predicted Probability')

# Show the plot
plt.title('Smoothed Predicted Probability Map in Tasmania')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(minimum_longitude, maximum_longitude)
plt.ylim(minimum_latitude, maximum_latitude)

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.show()

