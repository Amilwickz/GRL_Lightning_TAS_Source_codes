import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel-4-category\4_Binary_classification_dry_lightning_Aug_2004_to_Dec_2018.CSV')

# Select the relevant columns
selected_columns = ['Aspect', 'Elevation','FFDI', 'Slope', 'fuel load', 'RH', 'solar_rad',
                  'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction','FMI','SDI','1', '2','3', '4',  'Binary']
data = data[selected_columns]

# Split the data into features and target
X = data.drop('Binary', axis=1)
y = data['Binary']

# Use RandomUnderSampler to balance the class distribution
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define the parameter grid to search through
param_grid = {
   'n_estimators': [3, 5, 10, 50, 100, 200, 300, 500, 1000],
   'max_depth': [None, 10, 20],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4],
   'class_weight': [None, 'balanced']
}

# Create a Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Print headers
print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format("n_estimators", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Confusion Matrix"))

# Iterate over n_estimators values
for n_estimator in param_grid['n_estimators']:
    param_grid['n_estimators'] = [n_estimator]  # Set the current n_estimator
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {}".format(n_estimator, accuracy, precision, recall, f1, roc_auc, conf_matrix))

