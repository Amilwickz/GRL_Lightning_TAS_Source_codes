import pandas as pd

# Read the original CSV file
csv_file = r'C:/Users/amilaw/Desktop/Ignition Modelling project/Data\Data summary 3_Lightning/ZZ_Model_development/ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category/Random Forest/Predicted probability maps/2_Predicted_probability.CSV'
df = pd.read_csv(csv_file)

# Group by grid cells and calculate the mean 'Predicted Probability'
grouped_df = df.groupby(['Lower_Latitude', 'Higher_Latitude', 'Lower_Longitude', 'Higher_Longitude'])['Predicted Probability'].mean().reset_index()

# Rename the column to 'Mean Predicted Probability'
grouped_df.rename(columns={'Predicted Probability': 'Mean Predicted Probability'}, inplace=True)

# Save the result to a new CSV file
output_file = r'C:/Users/amilaw/Desktop/Ignition Modelling project/Data\Data summary 3_Lightning/ZZ_Model_development/ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category/Random Forest/Predicted probability maps/3_Predicted_probability_cell_each_1km.CSV'
grouped_df.to_csv(output_file, index=False)

print(f"Unique cells data saved to {output_file}")

