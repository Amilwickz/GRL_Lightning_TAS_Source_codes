import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel-4-category\4_Binary_classification_dry_lightning_Aug_2004_to_Dec_2018.CSV')

# Select the relevant columns
selected_columns = ['Aspect', 'Elevation', 'Slope', 'fuel load', '1', '2', '3', '4', 'FMI', 'SDI', 'RH', 'FFDI', 'solar_rad', 'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction', 'Binary']
data = data[selected_columns]

# Split the data into features and target
X = data.drop('Binary', axis=1)
y = data['Binary']

# Use RandomUnderSampler to balance the class distribution (optional, depending on your data)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Define feature labels for plotting
feature_labels = ['Aspect', 'Elevation', 'Slope', 'fuel load', '1', '2', '3', '4', 'FMI', 'SDI', 'RH', 'FFDI', 'solar_rad', 'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction']

# Plot partial dependence for each fuel type
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Define fuel types
fuel_types = ['1', '2', '3', '4']

for i, fuel_type in enumerate(fuel_types):
    ax = axes.flatten()[i]
    display = PartialDependenceDisplay.from_estimator(
        estimator=rf_model,
        X=X_test,
        features=[feature_labels.index('fuel load')],
        feature_names=feature_labels,
        response_method='predict_proba',
        ax=ax,
        kind='average',
    )
    display.plot(ax=display.axes_)  # Set ax=display.axes_ as recommended
    ax.set_title(f'Fuel Type {fuel_type}')
    ax.set_xlabel('Fuel Load')
    ax.set_ylabel('Partial Dependence')

plt.tight_layout()
plt.show()

