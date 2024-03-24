import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel-4-category\4_Binary_classification_dry_lightning_Aug_2004_to_Dec_2018.CSV')

# Select the relevant columns excluding 'Lat' and 'Lon'
selected_columns = ['Aspect', 'Elevation', 'Slope', 'fuel load','1', '2','3', '4','FMI','SDI', 'RH','FFDI', 'solar_rad','max_temp', 'vp_deficit', 'wind_speed', 'wind_direction', 'Binary']
data_model = data[selected_columns]

# Split the data into features and target
X = data_model.drop('Binary', axis=1)
y = data_model['Binary']

# Use RandomUnderSampler to balance the class distribution (optional, depending on your data)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Impute missing values with the mean for original X
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Impute missing values with the mean for X_resampled, X_train, and X_test
X_resampled = imputer.fit_transform(X_resampled)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Create a Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Calculate predicted probabilities for all records in the original data #This predict_proba comes in scikit-learn library in python.
predicted_probabilities = best_model.predict_proba(X)[:, 1]

# Add predicted probabilities and 'Lat' and 'Lon' columns to the original data
data['Predicted Probability'] = predicted_probabilities

# Save the updated data to a new CSV file
output_path = r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\ZZ_Model_development\ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category\Random Forest\Predicted probability maps\1_predicted_probability.csv'
data.to_csv(output_path, index=False)

print(f"Predicted probabilities saved to: {output_path}")

