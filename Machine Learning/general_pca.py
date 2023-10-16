import os
import numpy as np
from sklearn.decomposition import PCA
from useful_functions import get_legend_handles
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('lines', linewidth=2, color='black')
mpl.rc('font',family='serif',weight='bold')
mpl.rc('text',color='black')
mpl.rcParams['xtick.major.size']=15
mpl.rcParams['xtick.minor.size']=10
mpl.rcParams['xtick.labelsize']=42
mpl.rcParams['ytick.labelsize']=42
mpl.rcParams['ytick.major.size']=12
mpl.rcParams['ytick.minor.size']=6
mpl.rcParams['grid.linewidth']=1.5
mpl.rcParams['axes.labelsize']=52 
mpl.rcParams['legend.fontsize']=26
mpl.rcParams['figure.figsize'] = (16.0, 14.0)
cols = ['red','blue','green','k','cyan','purple','magenta','orange','yellow','indigo','violet','brown']
signs=['o','s','>','<','^','v','p','*','h','D','x','H','.']
ltype=['-','--','-.','-','--','-.','--','-','-.']

def perform_pca(data, n_components):
    """
    Perform PCA dimensionality reduction.

    Args:
        data (array): Input data for PCA.
        n_components (int): Number of components for PCA.

    Returns:
        components (array): PCA components.
        variances (array): Explained variances.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    components = pca.components_
    variances = pca.explained_variance_ratio_
    transformed_data = pca.transform(data)
    return components, variances, transformed_data

def plot_supervised_pca(transformed_data, labels_true, labels_pred, class_names, save_path=None):
    """
    Plot the data in the reduced PCA space with true and predicted labels.

    Args:
        transformed_data (array-like): Transformed data points.
        labels_true (array-like): True labels for the data points.
        labels_pred (array-like): Predicted labels for the data points.
        class_names (list): List of class names.
        save_path (str, optional): Directory to save the figures. Default is None.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    reduced_data = transformed_data

    for i in range(len(class_names)):
       mask_true = labels_true == i
       mask_pred = labels_pred == i
       true_marker = signs[i]  # Use the marker from signs list
       pred_marker = 'x'
       plt.scatter(reduced_data[mask_true, 0], reduced_data[mask_true, 1], c=cols[i], label=f'{class_names[i]} (True)', marker=true_marker)
       plt.scatter(reduced_data[mask_pred, 0], reduced_data[mask_pred, 1], c=cols[i], marker=pred_marker, s=100, label=f'{class_names[i]} (Predicted)')

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # Use get_legend_handles function to extract unique handles and labels
    handles, labels = get_legend_handles(ax)
    ax.legend(handles, labels)


    if save_path:
        fig.tight_layout()
        fig.savefig(save_path)
    else:
        plt.show()

""" 
The following is a general example you can use
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load sample dataset (Iris dataset)
data = load_iris()
X = data.data
y_true = data.target

# Define the number of components

n_components = 3

# Perform PCA
components, variances, transformed_data = perform_pca(X, n_components)

# Split the dataset into training and testing sets AFTER PCA
X_train, X_test, y_train, y_test = train_test_split(transformed_data, y_true, test_size=0.2, random_state=42)

# Train a simple classifier (Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)

# Predict on the test set
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_train == y_pred_train)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Visualize PCA
output_directory = 'pca_results'
class_names = ['Setosa', 'Versicolour', 'Virginica']  # Class names from the Iris dataset
# Plot supervised PCA with true and predicted labels
plot_supervised_pca(X_train, y_train, y_pred_train, class_names, save_path=output_directory+'/train_pca_plot.png')
plot_supervised_pca(X_test, y_test, y_pred, class_names, save_path=output_directory+'/test_pca_plot.png')