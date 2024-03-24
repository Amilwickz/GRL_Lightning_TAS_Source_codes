#@Amil-UTAS: Here I check the model using logistic regression resampling and I put the maximum iterations of 10000 for the convergence.
#Aslo, I checked the model performace with difference solvers such as 'liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'. Here I use only the independant variables as the predictors. No FFDI and FMI included
#This is modified to plot only the subplots in a one figure. No calculation has been changed from the script #2.Logistic regression-Resampling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

# Load the data from the CSV file
data = pd.read_csv(r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\Z_summing_data\Z_summing_data_including_Type_fuel-4-category\4_Binary_classification_dry_lightning_Aug_2004_to_Dec_2018.CSV')

# Select the relevant columns
selected_columns = ['Aspect', 'Elevation','FFDI', 'Slope', 'fuel load', 'RH', 'solar_rad',
                    'max_temp', 'vp_deficit', 'wind_speed', 'wind_direction', 'FMI','SDI','1', '2','3', '4','Binary']
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

# List of solvers to try
solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.8))


label_positions = [(0.7, 0.3), (0.7, 0.3), (0.7, 0.3), (0.7, 0.3), (0.7, 0.3)]

for idx, (solver, label_position) in enumerate(zip(solvers, label_positions), start=1):
    # Skip subplot (3, 2, 6)
    if idx == 6:
        continue

    # Train a Logistic Regression model with the current solver
    model = LogisticRegression(max_iter=10000, solver=solver)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Subplot configuration
    ax = plt.subplot(2, 3, idx)
    line = ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC (area={roc_auc:.2f})')
    line[0].set_label(f'ROC (area={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Add solver name inside the plot area
    ax.text(label_position[0], label_position[1], f'{solver}', ha='center', va='center', fontsize=10,
            transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))

    ax.legend(loc='lower right', fontsize=8)

# Remove subplot (3, 2, 6)
fig.delaxes(axes[1, 2])

# Set common x-axis and y-axis titles
fig.text(0.5, 0.02, 'False Positive Rate', ha='center', va='center', fontsize=12)
fig.text(0.02, 0.5, 'True Positive Rate', ha='center', va='center', rotation='vertical', fontsize=12)

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

