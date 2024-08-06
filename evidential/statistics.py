import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import linregress
from matplotlib.ticker import PercentFormatter

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import binned_statistic
from sklearn.metrics import precision_score, recall_score



def create_filtered_heatmap(data_dict):
    """
    Creates a heatmap using standard deviation data filtered based on a mask, with a controlled x-axis range
    to better fit the data distribution, and dynamic y-axis ranges and bin sizes based on the filtered data.

    Parameters:
    data_dict (dict): A dictionary containing two tensors, one for standard deviations ('std_dev')
                      and one for errors ('error_map'), and a 'mask' tensor for filtering.
    """
    # Convert tensors to numpy arrays and flatten them
    errors = data_dict['error_map'].detach().cpu().numpy().flatten()
    std_devs = data_dict['aleatoric_1'].detach().cpu().numpy().flatten()
    mask = data_dict['mask'].detach().cpu().numpy().flatten().astype(bool)

    # Apply the mask to filter the data
    filtered_errors = errors[mask]
    filtered_std_devs = std_devs[mask]

    if len(filtered_errors) == 0 or len(filtered_std_devs) == 0:
        print("No data points remain after filtering.")
        return

    # Compute optimal bin edges for both errors and standard deviations using the 'auto' strategy
    error_bins = np.histogram_bin_edges(filtered_errors, bins=1000)
    std_dev_bins = np.histogram_bin_edges(filtered_std_devs, bins=1000)

    # Create the heatmap
    plt.hist2d(filtered_errors, filtered_std_devs, bins=[error_bins, std_dev_bins], cmap=plt.cm.jet, density=True)
    plt.colorbar()

    t5 = max(filtered_errors)

    # Explicitly set x-axis limits to the range of your filtered error data
    plt.xlim(min(filtered_errors), 5)
    plt.ylim(0, 5)

    plt.xlabel('Error per Pixel (cm)')
    plt.ylabel('Aleatoric Uncertainty')
    plt.title('Aleatoric vs. Error')

    plt.show()


def create_pixelwise_heatmap_error(data_dict):
    """
    Creates a pixel-wise heatmap from a 2D array of errors, displaying each pixel in its original location,
    and shows a colorbar with scale (numbers) while keeping the image axis off.

    Parameters:
    data_dict (dict): A dictionary containing a 2D or 3D tensor/array 'error_map' and 'mask'.
                      The tensor/array should be 2D, or 3D with the first dimension being of size 1.
    """
    # Assuming 'error_map' might be 3D with the first dimension of size 1
    errors = data_dict['error_map'].detach().cpu().numpy()
    mask = data_dict['mask'].detach().cpu().numpy()

    # Handle the case where errors might have an extra first dimension of size 1
    if errors.ndim == 3 and errors.shape[0] == 1:
        errors = errors[0]  # Select the first element of the first dimension to make it 2D

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # Select the first element of the first dimension to make it 2D

    # Set masked values in errors to a specific value (e.g., -1)
    errors[mask == 0] = -1

    # Create a custom colormap that includes black for the masked value
    cmap = plt.cm.jet
    cmap.set_under(color='black')

    # Display the heatmap with custom colormap
    img = plt.imshow(errors, cmap=cmap, vmin=0, vmax=np.nanmax(errors))  # Set vmin to 0 to use the 'under' color
    plt.title('Prediction Error [mm]')

    # Create a colorbar with scale (numbers)
    cbar = plt.colorbar(img)
    # The colorbar will automatically have scale labels unless specified otherwise

    plt.axis('off')  # Keeps the image axis off
    plt.show()

def create_pixelwise_heatmap_alea_epis_single(data_dict):
    to_handle = ["aleatoric_1", "epistemic_1", "aleatoric_2", "epistemic_2"]

    for mode in to_handle:
        errors = data_dict['error_map'].detach().cpu().numpy()
        prediction = data_dict[mode].detach().cpu().numpy()
        mask = data_dict['mask'].detach().cpu().numpy()

        # Mask invalid values (NaNs)
        to_show = prediction[0]

        # Calculate valid min and max values for color scaling
        vmin = np.nanmin(to_show)
        vmax = np.nanmax(to_show)

        # Display the heatmap
        img = plt.imshow(to_show, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)  # Added vmin and vmax
        if mode == "aleatoric_1":
            plt.title("Relative: Aleatoric 1")
        if mode == "epistemic_1":
            plt.title("Relative: Epistemic 1")
        if mode == "aleatoric_2":
            plt.title("Relative: Aleatoric 2")
        if mode == "epistemic_2":
            plt.title("Relative: Epistemic 2")

        # Create a colorbar without scale (numbers)
        cbar = plt.colorbar(img)
        cbar.set_ticks([])  # Removes the ticks

        plt.axis('off')  # Optional: Remove the axis
        plt.show()


def create_pixelwise_heatmap_alea_epis(data_dict):
    to_handle = ["aleatoric_1", "epistemic_1"]

    for mode in to_handle:
        errors = data_dict['error_map'].detach().cpu().numpy()
        prediction = data_dict[mode].detach().cpu().numpy()
        mask = data_dict['mask'].detach().cpu().numpy()

        mask = mask[0]
        errors = (errors[0] + 0.5) * mask
        prediction = prediction[0] * mask

        to_show = prediction/errors * mask

        # Calculate valid min and max values for color scaling
        vmin = np.nanmin(to_show)
        vmax = np.nanmax(to_show)

        # Display the heatmap
        img = plt.imshow(to_show, cmap=plt.cm.jet, vmin=vmin, vmax=1000)  # Added vmin and vmax
        plt.title(mode + '/Error')

        # Create a colorbar without scale (numbers)
        cbar = plt.colorbar(img)
        cbar.set_ticks([])  # Removes the ticks

        plt.axis('off')  # Optional: Remove the axis
        plt.show()

def show_ref_image(data_dict):
    """
    Displays the original image stored under the 'ref_image' key in the provided dictionary,
    adjusting for images with batch and channel dimensions and rescaling from -1..1 to 0..1.

    Parameters:
    data_dict (dict): A dictionary containing an image under the key 'ref_image'.
                      The image can have a batch dimension, should be in the channel-first format,
                      and is normalized in the range -1 to 1.
    """
    # Extract the image from the dictionary
    image = data_dict['ref_img']

    # If the image is a tensor, convert it to a numpy array
    if hasattr(image, 'detach'):  # Check if 'image' is a PyTorch tensor
        image = image.detach().cpu().numpy()

    # Adjust for batch dimension and channel-first format
    if image.ndim == 4 and image.shape[0] == 1:
        # Select the first image in the batch and move the channel dimension to the last
        image = image[0].transpose(1, 2, 0)

    # Check if the image is grayscale (single channel)
    if image.shape[2] == 1:
        # If grayscale, remove the channel dimension
        image = image.squeeze(2)

    # Rescale the image from -1..1 to 0..1 for proper display
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)  # Ensure values are in the 0..1 range

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis ticks and labels
    plt.title('Reference Image')
    plt.show()

def read_tensors_from_pt_file(file_path):
    """
    Reads and returns the data stored in a .pt file, ensuring all tensors are on the CPU.
    """
    data = torch.load(file_path, map_location=torch.device('cpu'))
    return data

def analyze_uncertainties(folder_path):
    """
    Performs analysis of uncertainties and visualizations for three scenes stored in .pt files.
    """
    # List all .pt files in the specified folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

    # Dictionaries to hold data
    aleatoric_means = {}
    epistemic_means = {}

    def plot_mean_uncertainties(files):
        # Initialize a dictionary to store mean uncertainties
        mean_uncertainties = {}

        # Process each file and compute means using the mask
        for file in sorted(files):  # Sort files to maintain an order
            data = read_tensors_from_pt_file(file)
            print(data.keys())
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask
            mask = data['mask'].bool()
            aleatoric_mean_1 = data['aleatoric_1'][mask].mean().item()
            epistemic_mean_1 = data['epistemic_1'][mask].mean().item()
            aleatoric_mean_2 = data['aleatoric_2'][mask].mean().item()
            epistemic_mean_2 = data['epistemic_2'][mask].mean().item()

            # Store results in the dictionary
            mean_uncertainties[scene] = {
                'Aleatoric 1': aleatoric_mean_1,
                'Epistemic 1': epistemic_mean_1,
                'Aleatoric 2': aleatoric_mean_2,
                'Epistemic 2': epistemic_mean_2
            }

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.95  # Bar width
        space = 0 # Space between groups
        num_categories = len(next(iter(mean_uncertainties.values())))
        scenes = sorted(mean_uncertainties.keys())  # Sorted scenes
        scene_colors = plt.cm.get_cmap('viridis', len(scenes))  # Color map for scenes

        # Plot bars for each scene
        for i, scene in enumerate(scenes):
            offsets = np.arange(num_categories) * (len(scenes) + space) + i * width
            values = list(mean_uncertainties[scene].values())
            color = scene_colors(i)
            bars = ax.bar(offsets, values, width, label=scene, color=[color] * num_categories)

            # Annotate each bar with the value
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}',
                        va='bottom', ha='center', fontsize=10, color='black')

        # Adjust x-tick positions and labels
        tick_positions = np.arange(num_categories) * (len(scenes) + space) + width * (len(scenes) - 1) / 2
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(['Aleatoric 1', 'Epistemic 1', 'Aleatoric 2', 'Epistemic 2'])
        ax.set_ylabel('Mean Uncertainty [mm]')
        ax.set_title('Mean Uncertainties across Different Scenes')
        ax.legend(title='Scene')

        plt.tight_layout()
        plt.show()

    #plot_mean_uncertainties(files)

    def plot_mean_uncertainties_percentage(files):
        # Initialize a dictionary to store mean uncertainties
        mean_uncertainties = {}

        # Process each file and compute means using the mask
        for file in sorted(files):  # Sort files to maintain an order
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask
            mask = data['mask'].bool()
            aleatoric_mean_1 = data['aleatoric_1'][mask].mean().item()
            epistemic_mean_1 = data['epistemic_1'][mask].mean().item()
            aleatoric_mean_2 = data['aleatoric_2'][mask].mean().item()
            epistemic_mean_2 = data['epistemic_2'][mask].mean().item()

            # Store results in the dictionary
            mean_uncertainties[scene] = {
                'Aleatoric 1': aleatoric_mean_1,
                'Epistemic 1': epistemic_mean_1,
                'Aleatoric 2': aleatoric_mean_2,
                'Epistemic 2': epistemic_mean_2
            }

        # Determine the base values from the first scene for normalization
        first_scene = sorted(files)[0].split('/')[-1].split('.')[0]
        base_values = list(mean_uncertainties[first_scene].values())

        # Normalize other scenes' uncertainties relative to the first scene's values
        for scene in mean_uncertainties:
            mean_uncertainties[scene] = {
                key: (value / base_value) * 100 for key, value, base_value in
                zip(mean_uncertainties[scene].keys(), mean_uncertainties[scene].values(), base_values)
            }

        # Remove scene 1 from the plot data
        del mean_uncertainties[first_scene]

        # Prepare data for plotting
        scenes = sorted(mean_uncertainties.keys())
        categories = ['Aleatoric', 'Epistemic']
        methods = ['1', '2']
        num_categories = len(categories)
        num_methods = len(methods)

        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.4  # Bar width
        space = 0.1  # Space between groups

        # Define colors for methods 1 and 2
        colors = ['blue', 'green']

        # Calculate the number of bars per group
        num_bars_per_group = num_categories * num_methods

        # Create offsets for the bars
        offsets = np.arange(len(scenes) * num_bars_per_group) * (width + space)

        # Adjust offsets for specific bar positions
        increase_by = 0.1  # Increase value
        small_space = 0.2  # Smaller space between certain bars
        medium_space = 0.4  # Medium space between certain bars
        increase_position = [2, 6]  # Positions to increase by small_space
        large_increase_position = [4]  # Position to increase by medium_space

        for pos in increase_position:
            offsets[pos:] += small_space

        for pos in large_increase_position:
            offsets[pos:] += medium_space

        # Plot bars for each scene
        for i, scene in enumerate(scenes):
            values = list(mean_uncertainties[scene].values())
            for j in range(num_categories):
                for k in range(num_methods):
                    index = j * num_methods + k
                    bar = ax.bar(offsets[i * num_bars_per_group + index], values[index], width,
                                 label=f'{categories[j]}' if k == 0 and i == 0 else "", color=colors[k])

                    # Annotate each bar with the value
                    yval = bar[0].get_height()
                    ax.text(bar[0].get_x() + bar[0].get_width() / 2, yval, f'{yval:.2f}%',
                            va='bottom', ha='center', fontsize=10, color='black')

        # Adjust x-tick positions and labels
        tick_positions = (offsets.reshape(len(scenes), num_bars_per_group).mean(axis=1))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [f'Aleatoric                                  Epistemic\nScene {i + 2}' for i in range(len(scenes))],
            rotation=0)
        ax.set_ylabel('Relative Uncertainty (%)')
        ax.set_title('Relative Uncertainties (Scene 1 is 100%)')

        # Create custom legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0]), plt.Rectangle((0, 0), 1, 1, color=colors[1])]
        labels = ['Method 1', 'Method 2']
        ax.legend(handles, labels, title='Method')

        plt.tight_layout()
        plt.show()

    #plot_mean_uncertainties_percentage(files)

    def plot_density_uncertainties(files, use_mask=True):
        # Initialize lists to store uncertainties
        aleatoric_1 = []
        epistemic_1 = []
        aleatoric_2 = []
        epistemic_2 = []

        # Process each file and extract uncertainties
        for file in sorted(files):  # Sort files to maintain an order
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask if specified
            if use_mask:
                mask = data['mask'].bool()
            else:
                mask = torch.ones_like(data['mask']).bool()  # Use all data

            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            aleatoric_2.extend(data['aleatoric_2'][mask].tolist())
            epistemic_2.extend(data['epistemic_2'][mask].tolist())

        # Function to cut off the top 10% of values efficiently
        def filter_top_10_percent(values):
            cutoff_value = np.percentile(values, 90)
            return [v for v in values if v <= cutoff_value]

        # Apply the cutoff filter
        aleatoric_1 = filter_top_10_percent(aleatoric_1)
        epistemic_1 = filter_top_10_percent(epistemic_1)
        aleatoric_2 = filter_top_10_percent(aleatoric_2)
        epistemic_2 = filter_top_10_percent(epistemic_2)

        # Ensure non-negative values
        aleatoric_1 = [v for v in aleatoric_1 if v >= 0]
        epistemic_1 = [v for v in epistemic_1 if v >= 0]
        aleatoric_2 = [v for v in aleatoric_2 if v >= 0]
        epistemic_2 = [v for v in epistemic_2 if v >= 0]

        # Calculate percentages for histograms
        def calculate_percentages(data, bins):
            counts, bin_edges = np.histogram(data, bins=bins)
            percentages = (counts / counts.sum()) * 100
            return percentages, bin_edges

        # Aleatoric data with 1000 bins
        aleatoric_1_percent, aleatoric_1_bins = calculate_percentages(aleatoric_1, bins=1000)
        aleatoric_2_percent, aleatoric_2_bins = calculate_percentages(aleatoric_2, bins=1000)

        # Epistemic data with 1000 bins
        epistemic_1_percent, epistemic_1_bins = calculate_percentages(epistemic_1, bins=1000)
        epistemic_2_percent, epistemic_2_bins = calculate_percentages(epistemic_2, bins=1000)

        # Find the maximum density values for setting the y-axis limits
        max_aleatoric_1 = max(aleatoric_1_percent)
        max_aleatoric_2 = max(aleatoric_2_percent)
        max_epistemic_1 = max(epistemic_1_percent)
        max_epistemic_2 = max(epistemic_2_percent)

        # Find the 90th percentile for x-axis limits
        x_lim_aleatoric_1 = np.percentile(aleatoric_1, 90)
        x_lim_aleatoric_2 = np.percentile(aleatoric_2, 90)
        x_lim_epistemic_1 = np.percentile(epistemic_1, 90)
        x_lim_epistemic_2 = np.percentile(epistemic_2, 90)

        # Plotting histograms with different scales
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot aleatoric uncertainties
        width_1 = (aleatoric_1_bins[1] - aleatoric_1_bins[0])
        ax1.bar(aleatoric_1_bins[:-1], aleatoric_1_percent, width=width_1, align='edge', alpha=0.5,
                label='Aleatoric Method 1', color='blue')
        ax1.set_xlabel('Aleatoric Uncertainty [mm]', color='blue')
        ax1.set_ylabel('Density [%]', color='blue')
        ax1.tick_params(axis='x', labelcolor='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, max_aleatoric_1)
        ax1.set_xlim(left=0, right=x_lim_aleatoric_1)

        ax3 = ax1.twiny()
        ax4 = ax1.twinx()

        width_2 = (aleatoric_2_bins[1] - aleatoric_2_bins[0])
        ax3.bar(aleatoric_2_bins[:-1], aleatoric_2_percent, width=width_2, align='edge', alpha=0.5,
                label='Aleatoric Method 2', color='green')
        ax3.set_xlabel('Aleatoric Uncertainty [mm]', color='green')
        ax3.tick_params(axis='x', labelcolor='green')
        ax3.legend(loc='upper right')
        ax3.set_xlim(left=0, right=x_lim_aleatoric_2)

        # Scale the Method 2 values according to the right y-axis
        scale_factor_aleatoric = max_aleatoric_1 / max_aleatoric_2
        ax4.set_ylim(0, max_aleatoric_2)
        ax4.set_ylabel('Density [%]', color='green')
        ax4.tick_params(axis='y', labelcolor='green')

        for bar in ax3.containers[0]:
            bar.set_height(bar.get_height() * scale_factor_aleatoric)

        ax1.set_title(f'Aleatoric Uncertainties Histogram {"with" if use_mask else "without"} Mask')

        # Plot epistemic uncertainties
        width_1 = (epistemic_1_bins[1] - epistemic_1_bins[0])
        ax2.bar(epistemic_1_bins[:-1], epistemic_1_percent, width=width_1, align='edge', alpha=0.5,
                label='Epistemic Method 1', color='blue')
        ax2.set_xlabel('Epistemic Uncertainty [mm]', color='blue')
        ax2.set_ylabel('Density [%]', color='blue')
        ax2.tick_params(axis='x', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper left')
        ax2.set_ylim(0, max_epistemic_1)
        ax2.set_xlim(left=0, right=x_lim_epistemic_1)

        ax5 = ax2.twiny()
        ax6 = ax2.twinx()

        width_2 = (epistemic_2_bins[1] - epistemic_2_bins[0])
        ax5.bar(epistemic_2_bins[:-1], epistemic_2_percent, width=width_2, align='edge', alpha=0.5,
                label='Epistemic Method 2', color='green')
        ax5.set_xlabel('Epistemic Uncertainty [mm]', color='green')
        ax5.tick_params(axis='x', labelcolor='green')
        ax5.legend(loc='upper right')
        ax5.set_xlim(left=0, right=x_lim_epistemic_2)

        # Scale the Method 2 values according to the right y-axis
        scale_factor_epistemic = max_epistemic_1 / max_epistemic_2
        ax6.set_ylim(0, max_epistemic_2)
        ax6.set_ylabel('Density [%]', color='green')
        ax6.tick_params(axis='y', labelcolor='green')

        for bar in ax5.containers[0]:
            bar.set_height(bar.get_height() * scale_factor_epistemic)

        ax2.set_title(f'Epistemic Uncertainties Histogram {"with" if use_mask else "without"} Mask')

        plt.tight_layout()
        plt.show()

    # Plot density with mask
    #plot_density_uncertainties(files, use_mask=True)

    # Plot density without mask
    #plot_density_uncertainties(files, use_mask=False)

    def plot_uncertainty_heatmap(files, use_mask=True):
        # Initialize lists to store uncertainties and errors
        errors = []
        aleatoric_1 = []
        epistemic_1 = []
        aleatoric_2 = []
        epistemic_2 = []

        # Process each file and extract uncertainties and errors
        for file in sorted(files):  # Sort files to maintain an order
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask if specified
            if use_mask:
                mask = data['mask'].bool()
            else:
                mask = torch.ones_like(data['mask']).bool()  # Use all data

            errors.extend(data['error_map'][mask].tolist())
            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            aleatoric_2.extend(data['aleatoric_2'][mask].tolist())
            epistemic_2.extend(data['epistemic_2'][mask].tolist())

        # Combine the lists to preserve the associations
        combined_aleatoric_1 = list(zip(errors, aleatoric_1))
        combined_aleatoric_2 = list(zip(errors, aleatoric_2))
        combined_epistemic_1 = list(zip(errors, epistemic_1))
        combined_epistemic_2 = list(zip(errors, epistemic_2))

        # Function to cut off the top 10% of values
        def filter_top_10_percent(combined):
            errors = [item[0] for item in combined]
            uncertainties = [item[1] for item in combined]
            error_cutoff = np.percentile(errors, 90)
            uncertainty_cutoff = np.percentile(uncertainties, 90)
            return [item for item in combined if item[0] <= error_cutoff and item[1] <= uncertainty_cutoff]

        # Apply the cutoff filter to uncertainties and errors
        filtered_aleatoric_1 = filter_top_10_percent(combined_aleatoric_1)
        filtered_aleatoric_2 = filter_top_10_percent(combined_aleatoric_2)
        filtered_epistemic_1 = filter_top_10_percent(combined_epistemic_1)
        filtered_epistemic_2 = filter_top_10_percent(combined_epistemic_2)

        # Separate the filtered data
        errors_aleatoric_1 = [item[0] for item in filtered_aleatoric_1]
        aleatoric_1 = [item[1] for item in filtered_aleatoric_1]

        errors_aleatoric_2 = [item[0] for item in filtered_aleatoric_2]
        aleatoric_2 = [item[1] for item in filtered_aleatoric_2]

        errors_epistemic_1 = [item[0] for item in filtered_epistemic_1]
        epistemic_1 = [item[1] for item in filtered_epistemic_1]

        errors_epistemic_2 = [item[0] for item in filtered_epistemic_2]
        epistemic_2 = [item[1] for item in filtered_epistemic_2]

        # Create 2D histograms (heatmaps)
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))

        def plot_heatmap_with_regression(ax, x, y, title, xlabel, ylabel):
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax)

            # Compute linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            regression_line = slope * np.array(x) + intercept
            sorted_indices = np.argsort(x)
            ax.plot(np.array(x)[sorted_indices], regression_line[sorted_indices], color='red', linewidth=2)

        plot_heatmap_with_regression(ax[0, 0], errors_aleatoric_1, aleatoric_1, 'Aleatoric Uncertainty (Method 1)',
                                     'Error [cm]',
                                     'Aleatoric Uncertainty [mm]')
        plot_heatmap_with_regression(ax[0, 1], errors_aleatoric_2, aleatoric_2, 'Aleatoric Uncertainty (Method 2)',
                                     'Error [cm]',
                                     'Aleatoric Uncertainty [mm]')
        plot_heatmap_with_regression(ax[1, 0], errors_epistemic_1, epistemic_1, 'Epistemic Uncertainty (Method 1)',
                                     'Error [cm]',
                                     'Epistemic Uncertainty [mm]')
        plot_heatmap_with_regression(ax[1, 1], errors_epistemic_2, epistemic_2, 'Epistemic Uncertainty (Method 2)',
                                     'Error [cm]',
                                     'Epistemic Uncertainty [mm]')

        plt.tight_layout()
        plt.show()

    #plot_uncertainty_heatmap(files, use_mask=True)

    def plot_roc_pr_curves(files, T_e, T_u_list):

        def compute_roc_pr(errors, uncertainties, T_e, T_u):
            # Create labels based on error threshold T_e and uncertainty threshold T_u
            labels = (errors <= T_e) & (uncertainties <= T_u)

            # Check if there are any positive samples
            if np.sum(labels) == 0:
                return None, None, None, None, 0, 0

            # Compute ROC and PR curves
            fpr, tpr, _ = roc_curve(labels, uncertainties)  # No need to negate uncertainties
            precision, recall, _ = precision_recall_curve(labels, uncertainties)

            return fpr, tpr, precision, recall, auc(fpr, tpr), auc(recall, precision)


        # Initialize lists to store uncertainties and errors
        errors = []
        aleatoric_1 = []
        epistemic_1 = []
        aleatoric_2 = []
        epistemic_2 = []

        # Process each file and extract uncertainties and errors
        for file in sorted(files):  # Sort files to maintain an order
            data = read_tensors_from_pt_file(file)

            # Retrieve and apply the mask
            mask = data['mask'].bool()

            errors.extend(data['error_map'][mask].tolist())
            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            aleatoric_2.extend(data['aleatoric_2'][mask].tolist())
            epistemic_2.extend(data['epistemic_2'][mask].tolist())

        # Ensure non-negative values
        errors = np.array([v for v in errors if v >= 0])
        aleatoric_1 = np.array([v for v in aleatoric_1 if v >= 0])
        epistemic_1 = np.array([v for v in epistemic_1 if v >= 0])
        aleatoric_2 = np.array([v for v in aleatoric_2 if v >= 0])
        epistemic_2 = np.array([v for v in epistemic_2 if v >= 0])

        # Plot ROC and PR curves
        for T_u in T_u_list:
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            # Compute metrics for each method with the given T_u
            fpr_aleatoric_1, tpr_aleatoric_1, precision_aleatoric_1, recall_aleatoric_1, roc_auc_aleatoric_1, pr_auc_aleatoric_1 = compute_roc_pr(
                errors, aleatoric_1, T_e, T_u)
            fpr_aleatoric_2, tpr_aleatoric_2, precision_aleatoric_2, recall_aleatoric_2, roc_auc_aleatoric_2, pr_auc_aleatoric_2 = compute_roc_pr(
                errors, aleatoric_2, T_e, T_u)
            fpr_epistemic_1, tpr_epistemic_1, precision_epistemic_1, recall_epistemic_1, roc_auc_epistemic_1, pr_auc_epistemic_1 = compute_roc_pr(
                errors, epistemic_1, T_e, T_u)
            fpr_epistemic_2, tpr_epistemic_2, precision_epistemic_2, recall_epistemic_2, roc_auc_epistemic_2, pr_auc_epistemic_2 = compute_roc_pr(
                errors, epistemic_2, T_e, T_u)

            # Check if there are any valid metrics for plotting
            if fpr_aleatoric_1 is not None and fpr_aleatoric_2 is not None and fpr_epistemic_1 is not None and fpr_epistemic_2 is not None:
                # Plot ROC curves
                ax[0].plot(fpr_aleatoric_1, tpr_aleatoric_1,
                           label=f'Aleatoric Method 1 (AUC = {roc_auc_aleatoric_1:.2f})')
                ax[0].plot(fpr_aleatoric_2, tpr_aleatoric_2,
                           label=f'Aleatoric Method 2 (AUC = {roc_auc_aleatoric_2:.2f})')
                ax[0].plot(fpr_epistemic_1, tpr_epistemic_1,
                           label=f'Epistemic Method 1 (AUC = {roc_auc_epistemic_1:.2f})')
                ax[0].plot(fpr_epistemic_2, tpr_epistemic_2,
                           label=f'Epistemic Method 2 (AUC = {roc_auc_epistemic_2:.2f})')
                ax[0].plot([0, 1], [0, 1], 'k--')
                ax[0].set_xlabel('False Positive Rate')
                ax[0].set_ylabel('True Positive Rate')
                ax[0].set_title(f'ROC Curve (T_u={T_u})')
                ax[0].legend(loc='best')

                # Plot PR curves
                ax[1].plot(recall_aleatoric_1, precision_aleatoric_1,
                           label=f'Aleatoric Method 1 (AUC = {pr_auc_aleatoric_1:.2f})')
                ax[1].plot(recall_aleatoric_2, precision_aleatoric_2,
                           label=f'Aleatoric Method 2 (AUC = {pr_auc_aleatoric_2:.2f})')
                ax[1].plot(recall_epistemic_1, precision_epistemic_1,
                           label=f'Epistemic Method 1 (AUC = {pr_auc_epistemic_1:.2f})')
                ax[1].plot(recall_epistemic_2, precision_epistemic_2,
                           label=f'Epistemic Method 2 (AUC = {pr_auc_epistemic_2:.2f})')
                ax[1].set_xlabel('Recall')
                ax[1].set_ylabel('Precision')
                ax[1].set_title(f'Precision-Recall Curve (T_u={T_u})')
                ax[1].legend(loc='best')

                plt.tight_layout()
                plt.show()
            else:
                print(f"Skipping T_u={T_u} due to lack of positive samples.")

    T_e = 0.1  # Example error threshold
    T_u_list = [40, 50, 60]  # Example list of uncertainty thresholds
    #plot_roc_pr_curves(files, T_e, T_u_list)

    def plot_uncertainty_heatmap_with_precision_recall(files, threshold_values, save_folder):
        """
        Generate a pseudo precision-recall analysis for each scene based on specified error thresholds,
        filtering out zero errors and adjusting uncertainty threshold to include 90% of data.

        Parameters:
            files (list): List of file paths.
            threshold_values (list): List of thresholds for error classification.
            save_folder (str): Path to the folder where the plots will be saved.
        """

        def read_and_mask(file):
            data = read_tensors_from_pt_file(file)
            errors = data['error_map'].flatten()
            aleatoric_1 = data['aleatoric_1'].flatten()
            epistemic_1 = data['epistemic_1'].flatten()
            aleatoric_2 = data['aleatoric_2'].flatten()
            epistemic_2 = data['epistemic_2'].flatten()
            mask = data['mask'].bool().flatten()

            errors = errors[mask]
            aleatoric_1 = aleatoric_1[mask]
            epistemic_1 = epistemic_1[mask]
            aleatoric_2 = aleatoric_2[mask]
            epistemic_2 = epistemic_2[mask]

            return errors, aleatoric_1, epistemic_1, aleatoric_2, epistemic_2, os.path.basename(file).split('.')[0]

        def plot_precision_recall_combined(errors, uncertainty_1, uncertainty_2, threshold_values, uncertainty_type,
                                           scene, save_folder):
            for threshold in threshold_values:
                labels = errors < threshold  # True (1) if error is below threshold, False (0) otherwise

                # Sort data by increasing uncertainty
                sorted_indices_1 = np.argsort(uncertainty_1)
                sorted_labels_1 = labels[sorted_indices_1]
                sorted_uncertainty_1 = uncertainty_1[sorted_indices_1]

                sorted_indices_2 = np.argsort(uncertainty_2)
                sorted_labels_2 = labels[sorted_indices_2]
                sorted_uncertainty_2 = uncertainty_2[sorted_indices_2]

                # Determine the uncertainty threshold to include 90% of data
                index_90_percent_1 = int(0.9 * len(sorted_uncertainty_1))
                index_90_percent_2 = int(0.9 * len(sorted_uncertainty_2))
                min_uncertainty_90_percent_1 = sorted_uncertainty_1[index_90_percent_1]
                min_uncertainty_90_percent_2 = sorted_uncertainty_2[index_90_percent_2]

                # Calculate precision and recall up to the 90% uncertainty threshold
                cum_true_positives_1 = np.cumsum(sorted_labels_1)
                precision_1 = cum_true_positives_1 / np.arange(1, len(sorted_labels_1) + 1)
                recall_1 = cum_true_positives_1 / cum_true_positives_1[-1]

                cum_true_positives_2 = np.cumsum(sorted_labels_2)
                precision_2 = cum_true_positives_2 / np.arange(1, len(sorted_labels_2) + 1)
                recall_2 = cum_true_positives_2 / cum_true_positives_2[-1]

                # Create the plot with twin x-axes
                fig, ax1 = plt.subplots(figsize=(10, 6))

                ax2 = ax1.twiny()

                # Plot for Method 1
                ax1.plot(sorted_uncertainty_1[:index_90_percent_1 + 1], precision_1[:index_90_percent_1 + 1],
                         label='Precision Method 1', linestyle='--', color='blue')
                ax1.plot(sorted_uncertainty_1[:index_90_percent_1 + 1], recall_1[:index_90_percent_1 + 1],
                         label='Recall Method 1', linestyle='-', color='blue')
                ax1.set_xlabel('Uncertainty Method 1', color='blue')
                ax1.tick_params(axis='x', labelcolor='blue')

                # Plot for Method 2
                ax2.plot(sorted_uncertainty_2[:index_90_percent_2 + 1], precision_2[:index_90_percent_2 + 1],
                         label='Precision Method 2', linestyle='--', color='green')
                ax2.plot(sorted_uncertainty_2[:index_90_percent_2 + 1], recall_2[:index_90_percent_2 + 1],
                         label='Recall Method 2', linestyle='-', color='green')
                ax2.set_xlabel('Uncertainty Method 2', color='green')
                ax2.tick_params(axis='x', labelcolor='green')

                # Common settings
                ax1.set_ylabel('Precision / Recall')
                ax1.set_title(f'{uncertainty_type} | Scene {scene} | Threshold {threshold}mm')
                ax1.legend(loc='lower right')
                ax1.grid(True)

                # Add threshold text
                ax1.text(0.5, -0.2, 'Threshold', ha='center', va='center', transform=ax1.transAxes, fontsize=12)

                # Save the plot
                plot_filename = f"{uncertainty_type}_Scene_{scene}_Threshold_{threshold}mm.png"
                plot_path = os.path.join(save_folder, plot_filename)
                plt.savefig(plot_path)
                plt.close()

        for file in files:
            errors, aleatoric_1, epistemic_1, aleatoric_2, epistemic_2, scene = read_and_mask(file)

            # Plot for aleatoric uncertainties
            plot_precision_recall_combined(errors, aleatoric_1, aleatoric_2, threshold_values, 'Aleatoric', scene,
                                           save_folder)

            # Plot for epistemic uncertainties
            plot_precision_recall_combined(errors, epistemic_1, epistemic_2, threshold_values, 'Epistemic', scene,
                                           save_folder)

            # Plot for combined uncertainties
            combined_uncertainty_1 = aleatoric_1 + epistemic_1
            combined_uncertainty_2 = aleatoric_2 + epistemic_2
            plot_precision_recall_combined(errors, combined_uncertainty_1, combined_uncertainty_2, threshold_values,
                                           'Combined', scene, save_folder)

    save_folder = '/home/grannemann/Desktop/figures/precision_recall'
    #plot_uncertainty_heatmap_with_precision_recall(files, [2, 4, 6], save_folder=save_folder)


    def plot_error_distribution(files):
        """
        Plot error distribution for each scene separately in one plot.
        The x-axis represents the error, and the y-axis represents the cumulative percentage of values below each error threshold.

        Parameters:
            files (list): List of file paths.
        """

        def read_and_mask(file):
            data = read_tensors_from_pt_file(file)
            errors = data['error_map'].flatten()
            mask = data['mask'].bool().flatten()

            errors = errors[mask]

            return errors, os.path.basename(file).split('.')[0]

        plt.figure(figsize=(12, 8))

        for i, file in enumerate(files, start=1):
            errors, _ = read_and_mask(file)

            sorted_errors = np.sort(errors)
            cumulative_percentage = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

            plt.plot(sorted_errors, cumulative_percentage, label=str(i))

        plt.xlabel('Error [cm]')
        plt.ylabel('Cumulative Percentage')
        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        plt.title('Error Distribution per Scene')
        plt.legend(title='Scene')
        plt.xlim(0, 8)
        plt.grid(True)
        plt.show()

    #plot_error_distribution(files)


    def analyze_regression_error(files, use_mask=True):
        def calculate_regression_errors(x, y):
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.2, random_state=42)

            # Linear regression
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)

            # Polynomial regression (cubic)
            poly_features = PolynomialFeatures(degree=3)
            X_poly_train = poly_features.fit_transform(X_train)
            X_poly_test = poly_features.transform(X_test)

            poly_model = LinearRegression()
            poly_model.fit(X_poly_train, y_train)
            y_pred_poly = poly_model.predict(X_poly_test)

            # Calculate errors for Linear Regression
            mse_linear = mean_squared_error(y_test, y_pred_linear)
            rmse_linear = np.sqrt(mse_linear)
            mae_linear = mean_absolute_error(y_test, y_pred_linear)
            r2_linear = r2_score(y_test, y_pred_linear)

            # Calculate errors for Cubic Regression
            mse_poly = mean_squared_error(y_test, y_pred_poly)
            rmse_poly = np.sqrt(mse_poly)
            mae_poly = mean_absolute_error(y_test, y_pred_poly)
            r2_poly = r2_score(y_test, y_pred_poly)

            return {
                'Linear': {'MSE': mse_linear, 'RMSE': rmse_linear, 'MAE': mae_linear, 'R2': r2_linear},
                'Cubic': {'MSE': mse_poly, 'RMSE': rmse_poly, 'MAE': mae_poly, 'R2': r2_poly}
            }


        # Initialize lists to store uncertainties and errors
        aleatoric_1 = []
        epistemic_1 = []
        errors_aleatoric = []
        errors_epistemic = []

        # Process each file and extract uncertainties and errors
        for file in sorted(files):
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask if specified
            if use_mask:
                mask = data['mask'].bool()
            else:
                mask = torch.ones_like(data['mask']).bool()  # Use all data

            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            errors_aleatoric.extend(data['error_map'][mask].tolist())
            errors_epistemic.extend(data['error_map'][mask].tolist())

        # Convert lists to numpy arrays for further processing
        aleatoric_1 = np.array(aleatoric_1)
        epistemic_1 = np.array(epistemic_1)
        errors_aleatoric = np.array(errors_aleatoric)
        errors_epistemic = np.array(errors_epistemic)

        # Calculate regression errors for aleatoric and epistemic uncertainties
        aleatoric_errors = calculate_regression_errors(aleatoric_1, errors_aleatoric)
        epistemic_errors = calculate_regression_errors(epistemic_1, errors_epistemic)

        # Output the errors
        results = {
            "Linear Regression": {
                "Aleatoric": aleatoric_errors['Linear'],
                "Epistemic": epistemic_errors['Linear']
            },
            "Cubic Regression": {
                "Aleatoric": aleatoric_errors['Cubic'],
                "Epistemic": epistemic_errors['Cubic']
            }
        }

        print(results)

    #analyze_regression_error(files)

    def plot_error_over_std_vs_predicted_uncertainty(files, use_mask=True):
        # Initialize lists to store uncertainties and errors
        aleatoric_1 = []
        epistemic_1 = []
        errors_aleatoric = []
        errors_epistemic = []

        # Process each file and extract uncertainties and errors
        for file in sorted(files):
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask if specified
            if use_mask:
                mask = data['mask'].bool()
            else:
                mask = torch.ones_like(data['mask']).bool()  # Use all data

            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            errors_aleatoric.extend(data['error_map'][mask].tolist())
            errors_epistemic.extend(data['error_map'][mask].tolist())

        # Convert lists to numpy arrays for further processing
        aleatoric_1 = np.array(aleatoric_1)
        epistemic_1 = np.array(epistemic_1)
        errors_aleatoric = np.array(errors_aleatoric)
        errors_epistemic = np.array(errors_epistemic)

        # Calculate Error / Standard Deviation
        error_over_std_aleatoric = errors_aleatoric / np.std(errors_aleatoric)
        error_over_std_epistemic = errors_epistemic / np.std(errors_epistemic)

        # Plot Error / Standard Deviation vs Predicted Uncertainty
        plt.figure(figsize=(14, 7))

        # Plot for Aleatoric Uncertainty
        plt.subplot(1, 2, 1)
        plt.scatter(aleatoric_1, error_over_std_aleatoric, alpha=0.5, color='blue')
        plt.xlabel('Predicted Aleatoric Uncertainty')
        plt.ylabel('Error / Standard Deviation')
        plt.title('Aleatoric Uncertainty: Error / Std vs Predicted Uncertainty')

        # Plot for Epistemic Uncertainty
        plt.subplot(1, 2, 2)
        plt.scatter(epistemic_1, error_over_std_epistemic, alpha=0.5, color='green')
        plt.xlabel('Predicted Epistemic Uncertainty')
        plt.ylabel('Error / Standard Deviation')
        plt.title('Epistemic Uncertainty: Error / Std vs Predicted Uncertainty')

        plt.tight_layout()
        plt.show()

    #plot_error_over_std_vs_predicted_uncertainty(files)

    def calibration_plot(files, use_mask=True, bins=10):
        # Initialize lists to store uncertainties and errors
        aleatoric_1 = []
        epistemic_1 = []
        errors_aleatoric = []
        errors_epistemic = []

        # Process each file and extract uncertainties and errors
        for file in sorted(files):
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Retrieve and apply the mask if specified
            if use_mask:
                mask = data['mask'].bool()
            else:
                mask = torch.ones_like(data['mask']).bool()  # Use all data

            aleatoric_1.extend(data['aleatoric_1'][mask].tolist())
            epistemic_1.extend(data['epistemic_1'][mask].tolist())
            errors_aleatoric.extend(data['error_map'][mask].tolist())
            errors_epistemic.extend(data['error_map'][mask].tolist())

        # Convert lists to numpy arrays for further processing
        aleatoric_1 = np.array(aleatoric_1)
        epistemic_1 = np.array(epistemic_1)
        errors_aleatoric = np.array(errors_aleatoric)
        errors_epistemic = np.array(errors_epistemic)

        # Define a function to filter values outside the 5th and 95th percentiles
        def filter_extremes(predicted_uncertainty, errors):
            lower_bound = np.percentile(predicted_uncertainty, 2)
            upper_bound = np.percentile(predicted_uncertainty, 98)
            mask = (predicted_uncertainty >= lower_bound) & (predicted_uncertainty <= upper_bound)
            return predicted_uncertainty[mask], errors[mask]

        # Filter out extreme values for aleatoric and epistemic uncertainties
        aleatoric_1, errors_aleatoric = filter_extremes(aleatoric_1, errors_aleatoric)
        epistemic_1, errors_epistemic = filter_extremes(epistemic_1, errors_epistemic)

        def plot_calibration(predicted_uncertainty, errors, title, color):
            # Bin data
            bin_means, bin_edges, _ = binned_statistic(predicted_uncertainty, errors, statistic='mean', bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.plot(bin_centers, bin_means, marker='o', linestyle='-', color=color)
            plt.xlabel('Predicted Uncertainty [mm]')
            plt.ylabel('Mean Absolute Error [mm]')
            plt.title(title)

        # Plot Calibration for Aleatoric Uncertainty
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plot_calibration(aleatoric_1, errors_aleatoric, 'Aleatoric uncertainty Calibration', 'blue')

        # Plot Calibration for Epistemic Uncertainty
        plt.subplot(1, 2, 2)
        plot_calibration(epistemic_1, errors_epistemic, 'Epistemic Uncertainty Calibration', 'green')

        plt.tight_layout()
        plt.show()

    calibration_plot(files)

    def plot_error_distribution(files):
        """
        Plot error distribution for aleatoric and epistemic uncertainties.
        The x-axis represents the error, and the y-axis represents the cumulative percentage of values below each error threshold.

        Parameters:
            files (list): List of file paths.
        """

        def read_and_mask(file, uncertainty_type):
            data = read_tensors_from_pt_file(file)
            mask = data['mask'].bool().flatten()
            errors = data[uncertainty_type].flatten()

            errors = errors[mask]

            return errors, os.path.basename(file).split('.')[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        for i, file in enumerate(files, start=1):
            errors_aleatoric, scene_aleatoric = read_and_mask(file, 'aleatoric_1')
            errors_epistemic, scene_epistemic = read_and_mask(file, 'epistemic_1')

            # Aleatoric Errors
            sorted_errors_aleatoric = np.sort(errors_aleatoric)
            cumulative_percentage_aleatoric = np.arange(1, len(sorted_errors_aleatoric) + 1) / len(
                sorted_errors_aleatoric) * 100

            # Epistemic Errors
            sorted_errors_epistemic = np.sort(errors_epistemic)
            cumulative_percentage_epistemic = np.arange(1, len(sorted_errors_epistemic) + 1) / len(
                sorted_errors_epistemic) * 100

            axes[0].plot(sorted_errors_aleatoric, cumulative_percentage_aleatoric, label=f'{scene_aleatoric}')
            axes[0].set_xlabel('Predicted Aleatoric Uncertainty [mm]')
            axes[0].set_ylabel('Cumulative Percentage')
            axes[0].yaxis.set_major_formatter(PercentFormatter())
            axes[0].set_title('Aleatoric Uncertainty Distribution per Scene')
            axes[0].legend(title='Scene')
            axes[0].set_xlim(0, 50)
            axes[0].grid(True)

            axes[1].plot(sorted_errors_epistemic, cumulative_percentage_epistemic, label=f'{scene_epistemic}')
            axes[1].set_xlabel('Predicted Epistemic Uncertainty [mm]')
            axes[1].set_ylabel('Cumulative Percentage')
            axes[1].yaxis.set_major_formatter(PercentFormatter())
            axes[1].set_title('Epistemic Uncertainty Distribution per Scene')
            axes[1].legend(title='Scene')
            axes[1].set_xlim(0, 80)
            axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    #plot_error_distribution(files)

    def plot_uncertainty_heatmap_with_roc(files, threshold_values, save_folder):
        """
        Generate a pseudo ROC analysis for each scene based on specified error thresholds,
        filtering out zero errors and adjusting uncertainty threshold to include 90% of data.

        Parameters:
            files (list): List of file paths.
            threshold_values (list): List of thresholds for error classification.
            save_folder (str): Path to the folder where the plots will be saved.
        """

        def read_and_mask(file):
            data = read_tensors_from_pt_file(file)
            errors = data['error_map'].flatten()
            aleatoric_1 = data['aleatoric_1'].flatten()
            epistemic_1 = data['epistemic_1'].flatten()
            aleatoric_2 = data['aleatoric_2'].flatten()
            epistemic_2 = data['epistemic_2'].flatten()
            mask = data['mask'].bool().flatten()

            errors = errors[mask]
            aleatoric_1 = aleatoric_1[mask]
            epistemic_1 = epistemic_1[mask]
            aleatoric_2 = aleatoric_2[mask]
            epistemic_2 = epistemic_2[mask]

            return errors, aleatoric_1, epistemic_1, aleatoric_2, epistemic_2, os.path.basename(file).split('.')[0]

        def plot_roc_combined(errors, uncertainty_1, uncertainty_2, threshold_values, uncertainty_type, scene,
                              save_folder):
            for threshold in threshold_values:
                labels = errors < threshold  # True (1) if error is below threshold, False (0) otherwise

                # Calculate ROC for Method 1
                tpr_1, fpr_1, _ = roc_curve(labels, uncertainty_1)
                roc_auc_1 = auc(fpr_1, tpr_1)

                # Calculate ROC for Method 2
                tpr_2, fpr_2, _ = roc_curve(labels, uncertainty_2)
                roc_auc_2 = auc(fpr_2, tpr_2)

                # Create the plot with twin x-axes
                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Plot ROC for Method 1
                ax1.plot(fpr_1, tpr_1, label=f'AUC for Method 1: {roc_auc_1:.2f}', linestyle='-',
                         color='blue')
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate')
                ax1.set_title(f'{uncertainty_type} | Scene {scene} | Threshold {threshold} cm')
                ax1.legend(loc='lower right')
                ax1.grid(True)

                # Plot ROC for Method 2
                ax1.plot(fpr_2, tpr_2, label=f'AUC for Method 2: {roc_auc_2:.2f}', linestyle='-',
                         color='green')
                ax1.legend(loc='lower right')

                # Add red diagonal line from (0, 0) to (1, 1)
                ax1.plot([0, 1], [0, 1], color='red', linestyle='--')

                # Add threshold text
                ax1.text(0.5, -0.2, 'Threshold', ha='center', va='center', transform=ax1.transAxes, fontsize=12)

                # Save the plot
                plot_filename = f"{uncertainty_type}_Scene_{scene}_Threshold_{threshold}mm_ROC.png"
                plot_path = os.path.join(save_folder, plot_filename)
                plt.savefig(plot_path)
                plt.close()

        for file in files:
            errors, aleatoric_1, epistemic_1, aleatoric_2, epistemic_2, scene = read_and_mask(file)

            # Plot for aleatoric uncertainties
            plot_roc_combined(errors, aleatoric_1, aleatoric_2, threshold_values, 'Aleatoric', scene, save_folder)

            # Plot for epistemic uncertainties
            plot_roc_combined(errors, epistemic_1, epistemic_2, threshold_values, 'Epistemic', scene, save_folder)

            # Plot for combined uncertainties
            combined_uncertainty_1 = aleatoric_1 + epistemic_1
            combined_uncertainty_2 = aleatoric_2 + epistemic_2
            plot_roc_combined(errors, combined_uncertainty_1, combined_uncertainty_2, threshold_values, 'Combined',
                              scene, save_folder)

    save_folder = '/home/grannemann/Desktop/figures/roc'
    plot_uncertainty_heatmap_with_roc(files, [2, 4, 6], save_folder=save_folder)

    '''
    def plot_uncertainty_vs_error(errors, aleatoric, epistemic, scene):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist2d(errors, aleatoric, bins=30, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Error')
        plt.xlim(10)
        plt.ylabel('Aleatoric Uncertainty')
        plt.title(f'Aleatoric Uncertainty vs Error for {scene}')

        plt.subplot(1, 2, 2)
        plt.hist2d(errors, epistemic, bins=30, cmap='viridis')
        plt.colorbar()
        plt.xlim(10)
        plt.xlabel('Error')
        plt.ylabel('Epistemic Uncertainty')
        plt.title(f'Epistemic Uncertainty vs Error for {scene}')
        plt.show()

    def threshold_based_precision_recall(errors, uncertainty, threshold_values, scene):
        """
        Generate a pseudo precision-recall analysis based on specified error thresholds,
        filtering out zero errors and adjusting uncertainty threshold to include 90% of data.

        Parameters:
            errors (np.array): Flattened array of errors.
            uncertainty (np.array): Corresponding uncertainty values (can be aleatoric or epistemic).
            threshold_values (list): List of thresholds for error classification.
        """
        # Filter out where error is 0
        mask = errors != 0
        filtered_errors = errors[mask]
        filtered_uncertainty = uncertainty[mask]

        for threshold in threshold_values:
            labels = filtered_errors < threshold  # True (1) if error is below threshold, False (0) otherwise

            # Sort data by increasing uncertainty
            sorted_indices = np.argsort(filtered_uncertainty)
            sorted_labels = labels[sorted_indices]
            sorted_uncertainty = filtered_uncertainty[sorted_indices]

            # Determine the uncertainty threshold to include 90% of data
            index_90_percent = int(0.9 * len(sorted_uncertainty))
            min_uncertainty_90_percent = sorted_uncertainty[index_90_percent]

            # Calculate precision and recall up to the 90% uncertainty threshold
            cum_true_positives = np.cumsum(sorted_labels)
            precision = cum_true_positives / np.arange(1, len(sorted_labels) + 1)
            recall = cum_true_positives / cum_true_positives[-1]

            # Only plot up to the 90% uncertainty index
            plt.figure(figsize=(10, 6))
            plt.plot(sorted_uncertainty[:index_90_percent + 1], precision[:index_90_percent + 1], label='Precision',
                     linestyle='--')
            plt.plot(sorted_uncertainty[:index_90_percent + 1], recall[:index_90_percent + 1], label='Recall',
                     linestyle='-')
            plt.title(f'Precision-Recall vs. Uncertainty for {scene} at Threshold {threshold}')
            plt.xlabel('Uncertainty Threshold')
            plt.ylabel('Precision / Recall')
            plt.legend()
            plt.grid(True)
            plt.show()

    # Iterate through files and collect data
    for file in files:
        data = read_tensors_from_pt_file(file)
        scene = os.path.basename(file).split('.')[0]

        aleatoric_mean_1 = data['aleatoric_1'].mean().item()
        epistemic_mean_1 = data['epistemic_1'].mean().item()
        aleatoric_means[scene + '_1'] = aleatoric_mean_1
        epistemic_means[scene + '_1'] = epistemic_mean_1

        errors = data['error_map'].numpy().flatten()
        aleatoric_1 = data['aleatoric_1'].numpy().flatten()
        epistemic_1 = data['epistemic_1'].numpy().flatten()
        threshold_values = [5, 10, 15]  # Example thresholds

        plot_uncertainty_vs_error(errors, aleatoric_1, epistemic_1, scene)
        threshold_based_precision_recall(errors, aleatoric_1+epistemic_1, threshold_values, scene)


    # Plot means comparison
    fig, ax = plt.subplots()
    scenes = list(aleatoric_means.keys())
    ax.bar(scenes, [aleatoric_means[s] for s in scenes], color='b', label='Aleatoric')
    ax.bar(scenes, [epistemic_means[s] for s in scenes], color='r', alpha=0.5, label='Epistemic')
    ax.set_ylabel('Mean Uncertainty')
    ax.set_title('Comparison of Mean Uncertainties')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    def plot_uncertainties(files):
        for file in files:
            data = read_tensors_from_pt_file(file)
            scene = os.path.basename(file).split('.')[0]

            # Concatenate uncertainties from aleatoric_1 and epistemic_1
            uncertainties = np.concatenate(
                [data['aleatoric_1'].numpy().flatten(), data['epistemic_1'].numpy().flatten()])
            plt.figure()
            plt.hist(uncertainties, bins=np.linspace(0, 1000, 101), density=True, alpha=0.6, color='g')
            plt.title(f'Density of Uncertainties for {scene}: Method 1')
            plt.ylim(0, 0.2)
            #plt.xlim(0, 100)
            plt.xlabel('Uncertainty')
            plt.ylabel('Density')
            plt.show()

            # Concatenate uncertainties from aleatoric_2 and epistemic_2
            uncertainties = np.concatenate(
                [data['aleatoric_2'].numpy().flatten(), data['epistemic_2'].numpy().flatten()])
            plt.figure()
            plt.hist(uncertainties, bins=np.linspace(0, 1000, 101), density=True, alpha=0.6, color='g')
            plt.title(f'Density of Uncertainties for {scene}: Method 2')
            plt.ylim(0, 0.2)
            #plt.xlim(0, 100)
            plt.xlabel('Uncertainty')
            plt.ylabel('Density')
            plt.show()

    plot_uncertainties(files)

    # Comparing relative differences
    for i in range(len(files) - 1):
        for j in range(i + 1, len(files)):
            data_i = read_tensors_from_pt_file(files[i])
            data_j = read_tensors_from_pt_file(files[j])
            diff_aleatoric = np.abs(data_i['aleatoric_1'] - data_j['aleatoric_1']).mean().item()
            diff_epistemic = np.abs(data_i['epistemic_1'] - data_j['epistemic_1']).mean().item()
            print(
                f"Relative 1 difference between {files[i]} and {files[j]}: Aleatoric={diff_aleatoric}, Epistemic={diff_epistemic}")
        for j in range(i + 1, len(files)):
            data_i = read_tensors_from_pt_file(files[i])
            data_j = read_tensors_from_pt_file(files[j])
            diff_aleatoric = np.abs(data_i['aleatoric_2'] - data_j['aleatoric_2']).mean().item()
            diff_epistemic = np.abs(data_i['epistemic_2'] - data_j['epistemic_2']).mean().item()
            print(
                f"Relative 2 difference between {files[i]} and {files[j]}: Aleatoric={diff_aleatoric}, Epistemic={diff_epistemic}")

    # Create heatmap of expected uncertainty
    for file in files:
        data = read_tensors_from_pt_file(file)
        scene = os.path.basename(file).split('.')[0]

        # Squeeze the tensor to remove singleton dimensions
        aleatoric = data['aleatoric_1'].squeeze()

        plt.figure()
        plt.imshow(aleatoric, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Heatmap of Aleatoric Uncertainty for {scene}')
        plt.show()
        
        
        
    '''






def plot_scene_precision_recall(directory_path):
    """
    Generate a pseudo precision-recall analysis for each scene based on a fixed threshold of 4mm,
    filtering out the top and lowest 2 percent to exclude outliers.

    Parameters:
        directory_path (str): Path to the directory containing the data files.
    """

    def collect_files_from_directory(directory_path):
        files = []
        for root, dirs, file_names in os.walk(directory_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                pt_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pt')]
                if pt_files:
                    [files.append(file) for file in pt_files]
        return files

    def read_and_mask(file):
        data = torch.load(file)
        errors = data['error_map'].cpu().numpy().flatten()
        aleatoric_1 = data['aleatoric_1'].cpu().numpy().flatten()
        epistemic_1 = data['epistemic_1'].cpu().numpy().flatten()
        mask = data['mask'].cpu().numpy().astype(bool).flatten()

        errors = errors[mask]
        aleatoric_1 = aleatoric_1[mask]
        epistemic_1 = epistemic_1[mask]

        return errors, aleatoric_1, epistemic_1, os.path.basename(os.path.dirname(os.path.dirname(file)))

    def filter_percentiles(errors, uncertainty_1, lower_percentile=2, upper_percentile=98):
        lower_bound = np.percentile(uncertainty_1, lower_percentile)
        upper_bound = np.percentile(uncertainty_1, upper_percentile)
        mask = (uncertainty_1 >= lower_bound) & (uncertainty_1 <= upper_bound)
        return errors[mask], uncertainty_1[mask]

    def filter_lowest_percentiles(precision, recall, uncertainty, lower_percentile=0.5):
        lower_bound_precision = np.percentile(precision, lower_percentile)
        lower_bound_recall = np.percentile(recall, lower_percentile)
        mask = (precision >= lower_bound_precision) & (recall >= lower_bound_recall)
        return precision[mask], recall[mask], uncertainty[mask]

    def plot_precision_recall_combined(all_data, threshold, uncertainty_type):
        fig, ax1 = plt.subplots(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))

        for idx, (scene, data) in enumerate(all_data.items()):
            errors, uncertainty_1 = data

            # Sort data by increasing uncertainty
            sorted_indices_1 = np.argsort(uncertainty_1)
            sorted_errors = errors[sorted_indices_1]
            sorted_uncertainty_1 = uncertainty_1[sorted_indices_1]

            # Filter out the top and lowest 2 percent to exclude outliers
            sorted_errors, sorted_uncertainty_1 = filter_percentiles(sorted_errors, sorted_uncertainty_1, 2, 98)

            labels = sorted_errors < threshold  # True (1) if error is below threshold, False (0) otherwise

            # Calculate precision and recall
            cum_true_positives_1 = np.cumsum(labels)
            precision_1 = cum_true_positives_1 / np.arange(1, len(labels) + 1)
            recall_1 = cum_true_positives_1 / cum_true_positives_1[-1]

            # Filter out the lowest 0.5 percent of precision and recall values
            precision_1, recall_1, sorted_uncertainty_1 = filter_lowest_percentiles(precision_1, recall_1, sorted_uncertainty_1, 0.2)

            # Plot for each scene
            ax1.plot(sorted_uncertainty_1, precision_1, linestyle='--', label=f'{scene} - Precision', color=colors[idx])
            ax1.plot(sorted_uncertainty_1, recall_1, linestyle='-', label=f'{scene} - Recall', color=colors[idx])

        ax1.set_xlabel('Uncertainty [mm]')
        ax1.set_ylabel('Precision / Recall')
        ax1.set_title(f'{uncertainty_type} Precision-Recall Analysis (Threshold {threshold} mm)')
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        def get_key_number(key):
            return int(key.split('-')[0])
        sorted_unique_labels = dict(sorted(unique_labels.items(), key=lambda item: get_key_number(item[0])))
        ax1.legend(sorted_unique_labels.values(), sorted_unique_labels.keys(), loc='best')
        ax1.grid(True)
        plt.show()

    # Collect files
    files = collect_files_from_directory(directory_path)

    # Initialize dictionaries to hold data for each scene
    all_errors = {}
    all_aleatoric_1 = {}
    all_epistemic_1 = {}

    # Read and mask data from files
    for file in files:
        errors, aleatoric_1, epistemic_1, scene = read_and_mask(file)
        if scene not in all_errors:
            all_errors[scene] = []
            all_aleatoric_1[scene] = []
            all_epistemic_1[scene] = []
        all_errors[scene].append(errors)
        all_aleatoric_1[scene].append(aleatoric_1)
        all_epistemic_1[scene].append(epistemic_1)

    # Concatenate data for each scene
    for scene in all_errors.keys():
        all_errors[scene] = np.concatenate(all_errors[scene])
        all_aleatoric_1[scene] = np.concatenate(all_aleatoric_1[scene])
        all_epistemic_1[scene] = np.concatenate(all_epistemic_1[scene])

    fixed_threshold = 4  # Fixed threshold of 4mm

    # Plot for aleatoric uncertainties
    plot_precision_recall_combined({scene: (all_errors[scene], all_aleatoric_1[scene]) for scene in all_errors}, fixed_threshold, 'Aleatoric')

    # Plot for epistemic uncertainties
    plot_precision_recall_combined({scene: (all_errors[scene], all_epistemic_1[scene]) for scene in all_errors}, fixed_threshold, 'Epistemic')

    # Plot for combined uncertainties
    combined_uncertainty_1 = {scene: all_aleatoric_1[scene] + all_epistemic_1[scene] for scene in all_errors}
    plot_precision_recall_combined({scene: (all_errors[scene], combined_uncertainty_1[scene]) for scene in all_errors}, fixed_threshold, 'Combined')

if __name__ == "__main__":

    folder_path = "/home/grannemann/Desktop/3_test"
    #analyze_uncertainties(folder_path)

    # Example usage
    directory_path = '/home/grannemann/Desktop/Blickwinkel'
    plot_scene_precision_recall(directory_path)


    #file_path = "/home/grannemann/Desktop/3_test/1.pt"
    #results = read_tensors_from_pt_file(file_path)
    #print(results.keys())
    #create_filtered_heatmap(results)
    #create_pixelwise_heatmap_error(results)
    #create_pixelwise_heatmap_alea_epis(results)
    #create_pixelwise_heatmap_alea_epis_single(results)
    #show_ref_image(results)





    print("EoS")