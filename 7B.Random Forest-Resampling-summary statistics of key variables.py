import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel\Z_summing_data_including_Type_fuel_V2\4B_Binary_V2.CSV')

# Select key variables for summary statistics
selected_columns = ['Aspect', 'Elevation','Slope', 'fuel load', 'Fuel_type','FMI', 'SDI','FFDI', 'RH', 'solar_rad', 'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction', ]

# Filter data based on 'Binary' column
binary_one_data = data[data['Binary'] == 1]

# Calculate summary statistics for all data
summary_stats_all = data[selected_columns].describe()

# Calculate summary statistics for data where 'Binary' is 1
summary_stats_binary_one = binary_one_data[selected_columns].describe()

# Write summary statistics to text files
with open('7_Random_forest_key_variables_mean_std_all.txt', 'w') as file:
    file.write(summary_stats_all.to_string())

with open('7_Random_forest_key_variables_mean_std_binary_one.txt', 'w') as file:
    file.write(summary_stats_binary_one.to_string())

# Display summary statistics for all data
print("Summary Statistics for All Data:")
print(summary_stats_all)

# Display summary statistics for data where 'Binary' is 1
print("\nSummary Statistics for Data where 'Binary' is 1:")
print(summary_stats_binary_one)

# Plot histograms and kernel density plots for each variable with mean and std values for all data
plt.figure(figsize=(14.4, 6.4))
for i, column in enumerate(selected_columns, 1):
    plt.subplot(2, 7, i)
    sns.histplot(data[column], bins=20, color='navy', edgecolor='black', kde=True)
    plt.xlabel(column, fontsize=10, weight='bold')  # X-axis title
    plt.ylabel('Frequency', fontsize=10)  # Y-axis title
    mean_val = summary_stats_all.loc['mean', column]
    std_val = summary_stats_all.loc['std', column]
    plt.text(0.5, 1.1, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}', transform=plt.gca().transAxes, fontsize=10,
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Add a common y-axis title
plt.text(-0.5, 0.5, 'Frequency', transform=plt.gcf().transFigure, fontsize=12, rotation='vertical', va='center')

# Adjust layout to prevent overlap
plt.tight_layout()

# Remove grids
plt.subplots_adjust(hspace=0.5, wspace=0.5)
sns.despine(left=True, bottom=True)

plt.savefig('7B_Random_forest_key_variables_mean_std_all.png')
plt.show()

# Plot boxplots for each variable for all data
data[selected_columns].boxplot(figsize=(12.8, 6.4), vert=False, patch_artist=True)
plt.title('Boxplots of Key Variables for All Data', fontsize=12)
plt.savefig('7B_Random_forest_key_variables_boxplots_all.png')
plt.show()

# Plot histograms and kernel density plots for each variable with mean and std values for data where 'Binary' is 1
plt.figure(figsize=(12.8, 6.4))
for i, column in enumerate(selected_columns, 1):
    plt.subplot(2, 7, i)
    sns.histplot(binary_one_data[column], bins=50, color='skyblue', edgecolor='dodgerblue', kde=True)
    plt.xlabel(column, fontsize=10)  # X-axis title
    plt.ylabel('Frequency', fontsize=10)  # Y-axis title
    mean_val = summary_stats_binary_one.loc['mean', column]
    std_val = summary_stats_binary_one.loc['std', column]
    plt.text(0.5, 1.1, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', transform=plt.gca().transAxes, fontsize=8,
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Add a common y-axis title
plt.text(-0.5, 0.5, 'Frequency', transform=plt.gcf().transFigure, fontsize=12, rotation='vertical', va='center')

# Adjust layout to prevent overlap
plt.tight_layout()

# Remove grids
plt.subplots_adjust(hspace=0.5, wspace=0.5)
sns.despine(left=True, bottom=True)

plt.savefig('7B_Random_forest_key_variables_mean_std_binary_one.png')
plt.show()

# Plot boxplots for each variable for data where 'Binary' is 1
binary_one_data[selected_columns].boxplot(figsize=(12.8, 6.4), vert=False, patch_artist=True)
plt.title('Boxplots of Key Variables for Data where \'Binary\' is 1', fontsize=12)
plt.savefig('7B_Random_forest_key_variables_boxplots_binary_one.png')
plt.show()

