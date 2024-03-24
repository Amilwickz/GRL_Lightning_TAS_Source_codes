import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel\Z_summing_data_including_Type_fuel_V2\4_Binary_classification_dry_lightning_Aug_2004_to_Dec_2018.CSV')

# Select the relevant columns
selected_columns = ['Aspect', 'Elevation', 'Slope', 'fuel load','FMI','Fuel_type','SDI', 'RH','FFDI', 'solar_rad',
                    'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction', 'Binary']
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

# Feature Importance Plot with Different Colors
feature_importances = best_model.feature_importances_
# Define custom feature names
custom_feature_names = {
    'Aspect': 'Aspect', 'Elevation': 'Elevation', 'Slope': 'Slope', 'fuel load': 'Fuel load', 'Fuel_type':'Fuel type',
    'FMI': 'FMI', 'SDI': 'SDI', 'RH': 'RH', 'FFDI': 'FFDI', 'solar_rad': 'Solar rad',
    'max_temp': 'Max temp', 'vp_deficit': 'Vapour press',
    'wind_speed': 'Wind speed', 'wind_direction': 'Wind direction'
}
feature_names = [custom_feature_names[col] for col in X.columns]

# Create a dictionary to store feature importance values for each variable
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Sort feature importances and feature names in ascending order
sorted_indices = feature_importances.argsort()
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]


custom_colors = {
    'Aspect': '#d47264', 'Elevation': '#d47264', 'Slope': '#d47264',
    'Fuel load': '#59a89c', 'FMI': '#59a89c','Fuel type': '#59a89c',
    'SDI': '#2066a8', 'RH': '#2066a8', 'FFDI': '#2066a8',
    'Solar rad': '#2066a8', 'Max temp': '#2066a8',
    'Vapour press': '#2066a8', 'Wind speed': '#2066a8', 'Wind direction': '#2066a8'
}





# Assign colors to features in sorted order
colors = [custom_colors[feature] for feature in sorted_feature_names]


# Plot feature importance bars in ascending order with custom colors
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_feature_importances, color=colors, edgecolor='black', linewidth=1)
plt.xlabel('Feature Importance', fontsize=16)
plt.show()


# Print feature importance values in the terminal
print("\nFeature Importance Values:")
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

# Correlation Heatmap without 'Binary'
plt.figure(figsize=(12, 8))
columns_to_exclude = ['Binary']
correlation_columns = [col for col in selected_columns if col not in columns_to_exclude]
correlation_matrix = data[correlation_columns].corr()

# Create custom feature names for the axis labels
custom_feature_names = {
    'Aspect': 'Aspect', 'Elevation': 'Elevation', 'Slope': 'Slope', 'fuel load': 'Fuel load','Fuel_type':'Fuel type',
    'FMI': 'FMI', 'SDI': 'SDI', 'RH': 'RH', 'FFDI': 'FFDI', 'solar_rad': 'Solar rad',
    'max_temp': 'Max temp', 'vp_deficit': 'Vapour press',
    'wind_speed': 'Wind speed', 'wind_direction': 'Wind direction'
}

# Replace column names with custom names
correlation_columns_custom_names = [custom_feature_names[col] for col in correlation_columns]

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=correlation_columns_custom_names, yticklabels=correlation_columns_custom_names, cbar_kws={'orientation': 'vertical'}, annot_kws={"size": 12})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.show()



# Confusion Matrix
y_pred = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Not Fire', 'Fire'], yticklabels=['Not Fire', 'Fire'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()




# Receiver Operating Characteristic (ROC) Curve
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Model Evaluation Metrics Table
metrics_dict = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC-ROC': roc_auc
}
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Score'])
print("\nModel Evaluation Metrics:")
print(metrics_df)

# Write metrics to a text file
metrics_df.to_csv('8_Random_forest_key_variables_mean_std_only heat map.txt', sep='\t')
