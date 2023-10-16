import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#---------- General Plotting Parameters I use -------#
# Can be copied to the main file which call these functions #
import matplotlib as mpl
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
	
def fit_power_law(x, y, start_idx, end_idx):
    """
    Fit a power-law function of the form y = A * x^b to a subset of data.

    Parameters:
        x (numpy.ndarray): Array of x-values.
        y (numpy.ndarray): Array of corresponding y-values.
        start_idx (int): Starting index for the subset of data.
        end_idx (int): Ending index for the subset of data.

    Returns:
        x_subset (numpy.ndarray): Subset of x-values.
        y_fit (numpy.ndarray): Fitted y-values.
        label (str): Label for the fit.
        coefficients (numpy.ndarray): Coefficients [A, b] of the power-law fit.
    """
    x_subset = x[start_idx:end_idx]
    y_subset = y[start_idx:end_idx]

    coefficients = np.polyfit(np.log10(x_subset), np.log10(y_subset), 1)
    A, b = 10**coefficients[1], coefficients[0]

    y_fit = A * x_subset**b
    label = f'Fit: $A \\cdot x^b$, b = {b:.2f}'

    return x_subset, y_fit, label, coefficients
	
def find_nearest_value(array, target_value):
    """
    Find the nearest value to the target value in an array.

    Parameters:
        array (numpy.ndarray): Array of values.
        target_value (float): Value to find the nearest value to.

    Returns:
        nearest_idx (int): Index of the nearest value.
        nearest_value (float): Nearest value in the array.
    """
    nearest_idx = np.abs(array - target_value).argmin()
    nearest_value = array[nearest_idx]
    return nearest_idx, nearest_value

def get_legend_handles(ax):
    """
    Extracts legend handles and labels from a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object from which to extract handles and labels.

    Returns:
        list: List of legend handles.
        list: List of legend labels.
    """
    handle_list, label_list = [], []
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    return handle_list, label_list

def customize_plot(ax, x_label, y_label, frame=True, legend=True):
    """
    Customize the appearance of a matplotlib plot.

    Args:
        ax (matplotlib.axes.Axes): The axis object to be customized.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        frame (bool, optional): Whether to display the frame. Default is True.
        legend (bool, optional): Whether to include a legend. Default is True.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', direction='out', length=15, width=2, color='k', pad=15, labelcolor='k')
    ax.set_xlabel(x_label, fontweight='bold', labelpad=25)
    ax.set_ylabel(y_label, fontweight='bold', labelpad=25)
    
    if not frame:
        ax.set_frame_on(False)
    if not legend:
        ax.legend().set_visible(False)

def customize_scatter_plot(ax, cols, labels):
    """
    Customize the appearance of a scatter plot.

    Args:
        ax (matplotlib.axes.Axes): The axis object to be customized.
        cols (list): List of colors corresponding to the scatter points.
        labels (list): Labels corresponding to the scatter points.
    """
    for x, z in zip(ax.get_children()[1:], labels):
        x.set_color(cols[z])

def customize_2d_plot(ax, cols, labels):
    """
    Customize the appearance of a 2D plot.

    Args:
        ax (matplotlib.axes.Axes): The axis object to be customized.
        cols (list): List of colors corresponding to the plot lines.
        labels (list): Labels corresponding to the plot lines.
    """
    for x, z in zip(ax.get_lines(), labels):
        x.set_color(cols[z])

def scatter_3d_plot(data_files, labels, x_label, y_label, z_label, columns):
    """
    Create a 3D scatter plot using data from multiple files.

    Args:
        data_files (list of str): List of file paths containing data for the plot.
            Each file should be formatted as follows: 
            - Column 1: x-data
            - Column 2: y-data
            - Column 3: z-data
        labels (list of str): Labels corresponding to the data files.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        z_label (str): Label for the z-axis.
        columns (int): Number of columns in the legend.

    Returns:
        matplotlib.figure.Figure: The created figure.
        matplotlib.axes._subplots.Axes3DSubplot: The 3D axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i, file_path in enumerate(data_files):
        data = np.loadtxt(file_path)
        x_data = data[0]
        y_data = data[1]
        z_data = data[2]
        
        label = labels[i]
        
        ax.scatter(x_data, y_data, z_data, marker=signs[i], color=cols[i], s=350, label=label)

    handle_list, label_list = get_legend_handles(ax)
    ax.legend(handle_list, label_list, loc='best', fancybox=True, framealpha=0.5, ncol=columns)    
    customize_plot(ax, x_label, y_label)
    ax.xaxis._axinfo['label']['space_factor'] = 8.5
    ax.yaxis._axinfo['label']['space_factor'] = 8.5
    ax.zaxis._axinfo['label']['space_factor'] = 8.5

    return fig, ax

def make_2dplot(data_files, idx1, idx2, labels, x_label, y_label, columns):
    """
    Create a 2D plot using data from multiple files.

    Args:
        data_files (list of str): List of file paths containing data for the plot.
            Each file should be formatted as follows: 
            - Column 1: x-data
            - Column 2: y-data
        idx1 (int): Column index for x-data in the files.
        idx2 (int): Column index for y-data in the files.
        labels (list of str): Labels corresponding to the data files.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        columns (int): Number of columns in the legend.

    Returns:
        matplotlib.figure.Figure: The created figure.
        matplotlib.axes._subplots.AxesSubplot: The 2D axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i, file_path in enumerate(data_files):
        data = np.loadtxt(file_path)
        x_data = data[idx1]
        y_data = data[idx2]
        
        label = labels[i]
        
        ax.plot(x_data, y_data, marker=signs[i], color=cols[i],  label=label)   

    handle_list, label_list = get_legend_handles(ax)  # Get the legend handles
    ax.legend(handle_list, label_list, loc='best', fancybox=True, framealpha=0.5, ncol=columns)
	
    customize_plot(ax, x_label, y_label)
	
    return fig, ax

def clustering_plot(data, cluster_model, predictions, x_label, y_label):
    """
    Create a clustering plot.

    Args:
        data (numpy.ndarray): Data points with shape (n_samples, n_features).
        cluster_model: The clustering model (e.g., KMeans) already fitted on the data.
        predictions (numpy.ndarray): Predicted cluster labels for each data point.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.

    Returns:
        matplotlib.figure.Figure: The created figure.
        matplotlib.axes._subplots.AxesSubplot: The axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.set_cmap('viridis')

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    h = .01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = cluster_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Paired,
              aspect='auto', origin='lower')

    unique_labels = np.unique(predictions)
    label_color_dict = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for label in unique_labels:
        cluster_data = data[predictions == label]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=400, c=[label_color_dict[label]], label=f'Cluster {label}')

    centers = cluster_model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], s=450, c='k', marker='*', alpha=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label, fontweight='bold', labelpad=15)
    ax.set_ylabel(y_label, fontweight='bold', labelpad=15)

    customize_plot(ax, x_label, y_label)

    return fig, ax

