#@Amil-UTAS
import matplotlib.pyplot as plt
import seaborn as sns

#The best performing solvers and the numerators for each model are mentioned.
#In the logistic regression- lbfgs solver
#In Random forest- trees number = 100 (default)
#In Gradient boosting -Number of estimators =100 and maximum depth=5


import matplotlib.pyplot as plt
import seaborn as sns

# Data for bar graphs
data_bar = {
    "Logistic regression": {
        "Accuracy": 0.89,
        "Precision": 0.88,
        "Recall": 0.90,
        "F1 Score": 0.89,
        "ROC-AUC": 0.94
    },
    "Random forest": {
        "Accuracy": 0.97,
        "Precision": 0.97,
        "Recall": 0.97,
        "F1 Score": 0.97,
        "ROC-AUC": 0.99
    },
    "Gradient boosting": {
        "Accuracy": 0.97,
        "Precision": 0.97,
        "Recall": 0.96,
        "F1 Score": 0.96,
        "ROC-AUC": 0.98
    }
}

# Data for confusion matrices
conf_matrix_lr = [[67, 9], [7, 64]]
conf_matrix_rf = [[74, 2], [2, 69]]
conf_matrix_gb = [[74, 5], [3, 68]]

# Plot bar graphs and confusion matrices
for fig_num, (data, conf_matrix) in enumerate(zip(data_bar.items(), [conf_matrix_lr, conf_matrix_rf, conf_matrix_gb]), start=1):
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    title, scores = data

    # Plot bar graph
    ax = axs[0]
    bars = ax.bar(scores.keys(), scores.values(), color=['#2066a8', '#a559aa', '#59a89c', '#f0c571', '#e02b35'], width=0.9, align='center')
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel('Score', fontsize=16)
    ax.tick_params(axis='x', rotation=45, width=2, labelsize=12)  # Adjusting font size of score keys
    ax.tick_params(axis='y', width=1)
    ax.margins(x=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Plot confusion matrix
    ax = axs[1]
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'], ax=ax, linewidths=1, linecolor='black')
    ax.set_xlabel('Predicted', fontsize=16)  # Adjusting font size of x-axis label
    ax.set_ylabel('Actual', fontsize=16)  # Adjusting font size of y-axis label

    # Save figure
    fig.savefig(f'figure_{fig_num}.png', bbox_inches='tight')

plt.show()

