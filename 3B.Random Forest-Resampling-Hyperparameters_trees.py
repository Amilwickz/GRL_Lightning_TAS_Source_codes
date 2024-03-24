import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
excel_path = r'C:\Users\amilaw\Desktop\Ignition Modelling project\Data\Data summary 3_Lightning\ZZ_Model_development\ZZ_Model_development_including_FMI_FFDI_SDI_Fuel_type-4-category\Random Forest\3_Random Forest-Resampling-Hyperparameters_trees.xlsx'
df = pd.read_excel(excel_path)

# Extract data
n_estimators = df['n_estimators (trees)']
accuracy = df['Accuracy']
precision = df['Precision']
recall = df['Recall']
f1_score = df['F1 Score']
roc_auc = df['ROC-AUC']

# Plotting
fig, ax = plt.subplots(figsize=(8.4, 4.8))
fig.patch.set_linewidth(2)  # Set border thickness

# Plot lines with specified options
ax.plot(n_estimators, accuracy, label='Accuracy', linestyle='-', linewidth=2, marker='o', markersize=4, color='dodgerblue', markerfacecolor='dodgerblue')
ax.plot(n_estimators, precision, label='Precision', linestyle='--', linewidth=2, marker='o', markersize=4, color='limegreen', markerfacecolor='limegreen')
ax.plot(n_estimators, recall, label='Recall', linestyle='-.', linewidth=2, marker='o', markersize=4, color='crimson', markerfacecolor='crimson')
ax.plot(n_estimators, f1_score, label='F1 Score', linestyle=':', linewidth=2, marker='o', markersize=4, color='purple', markerfacecolor='purple')
ax.plot(n_estimators, roc_auc, label='ROC-AUC', linestyle='-', linewidth=2, marker='o', markersize=4, color='orange', markerfacecolor='orange')

# Customize the plot
ax.set_xlabel('n_estimators (number of trees)')
ax.set_ylabel('Accuracy, Precision, Recall, F1 Score, ROC-AUC', fontsize=10)

# Display the legend in a single row
ax.legend(loc='lower center', bbox_to_anchor=(0.525, 0.005), ncol=5)

# Show only the x-axis ticks given by n_estimators
ax.set_xticks(n_estimators)

ax.grid(False)

# Show the plot
plt.show()

