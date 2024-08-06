import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the paths to the subfolders
dropout_folder = '/home/grannemann/PycharmProjects/AA-RMVSNet/outputs_dtu/dropout'
evidential_folder = '/home/grannemann/PycharmProjects/AA-RMVSNet/outputs_dtu/evidential'
output_folder = '/home/grannemann/PycharmProjects/AA-RMVSNet/output_plots'

os.makedirs(output_folder, exist_ok=True)

# Function to load a .pkl file and return its contents
def load_pkl_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

# Function to normalize image to 0-255
def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img_normalized = 255 * (img - img_min) / (img_max - img_min)
    return img_normalized.astype(np.uint8)

# Function to apply mask and set masked areas to black
def apply_mask(image, mask):
    masked_image = np.ma.masked_where(mask == 0, image)
    return masked_image

# Get the list of files in both directories and find the common files
dropout_files = set(f for f in os.listdir(dropout_folder) if f.endswith('.pkl'))
evidential_files = set(f for f in os.listdir(evidential_folder) if f.endswith('.pkl'))
common_files = sorted(list(dropout_files.intersection(evidential_files)))[:33]

for i, file in enumerate(common_files):
    dropout_file = os.path.join(dropout_folder, file)
    evidential_file = os.path.join(evidential_folder, file)

    dropout_depth_est_list = []
    dropout_error_map_list = []
    mask = None

    # Load dropout data
    dropout_data = load_pkl_file(dropout_file)
    for item in dropout_data['image_outputs_list']:
        dropout_depth_est_list.append(item['depth_est'])
        dropout_error_map_list.append(item['error_map'])
        if mask is None:
            mask = item['mask']

    # Calculate the variance of depth_est tensors
    depth_est_stack = torch.stack(dropout_depth_est_list, dim=0)
    depth_est_variance = torch.var(depth_est_stack, dim=0, unbiased=False)

    # Calculate the mean of all dropout error maps
    dropout_error_map_stack = torch.stack(dropout_error_map_list, dim=0)
    dropout_error_map_mean = torch.mean(dropout_error_map_stack, dim=0)

    # Load evidential data
    evidential_data = load_pkl_file(evidential_file)
    alea_1 = evidential_data['image_outputs_list'][0]['alea_1']
    epis_1 = evidential_data['image_outputs_list'][0]['epis_1']
    evidential_ref_img = evidential_data['image_outputs_list'][0]['ref_img']
    evidential_error_map = evidential_data['image_outputs_list'][0]['error_map']

    # Convert tensors to numpy arrays for visualization
    depth_est_variance_np = depth_est_variance.cpu().numpy().squeeze()
    alea_1_np = alea_1.cpu().numpy().squeeze()
    epis_1_np = epis_1.cpu().numpy().squeeze()
    evidential_ref_img_np = evidential_ref_img.cpu().numpy().squeeze().transpose(1, 2, 0)  # Transpose to get height x width x channels
    dropout_error_map_mean_np = dropout_error_map_mean.cpu().numpy().squeeze()
    evidential_error_map_np = evidential_error_map.cpu().numpy().squeeze()
    mask_np = mask.cpu().numpy().squeeze()

    # Normalize images to 0-255
    evidential_ref_img_np_normalized = normalize_image(evidential_ref_img_np)

    # Clip the top 2% of variance and error map values
    variance_98th_percentile = np.percentile(depth_est_variance_np, 98)
    depth_est_variance_clipped_np = np.clip(depth_est_variance_np, None, variance_98th_percentile)

    error_98th_percentile_dropout = np.percentile(dropout_error_map_mean_np, 98)
    dropout_error_map_mean_clipped_np = np.clip(dropout_error_map_mean_np, None, error_98th_percentile_dropout)

    error_98th_percentile_evidential = np.percentile(evidential_error_map_np, 98)
    evidential_error_map_clipped_np = np.clip(evidential_error_map_np, None, error_98th_percentile_evidential)

    # Apply masks to error maps
    dropout_error_map_mean_masked = apply_mask(dropout_error_map_mean_clipped_np, mask_np)
    evidential_error_map_masked = apply_mask(evidential_error_map_clipped_np, mask_np)

    # Create a custom colormap that includes black for the masked value
    cmap = plt.cm.jet
    cmap.set_bad(color='black')

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # Plot evidential reference image
    axs[0, 0].imshow(evidential_ref_img_np_normalized)
    axs[0, 0].set_title('Reference Image')
    axs[0, 0].axis('off')  # Hide axis for the image

    # Plot variance of depth_est without masking
    im1 = axs[0, 1].imshow(depth_est_variance_clipped_np, cmap=cmap, vmin=0, interpolation='none')
    axs[0, 1].set_title('Variance of MCD [mm]')
    fig.colorbar(im1, ax=axs[0, 1])
    axs[0, 1].axis('off')  # Hide axis

    # Plot alea_1
    im2 = axs[1, 0].imshow(alea_1_np, cmap=cmap, vmin=0, interpolation='none')
    axs[1, 0].set_title('Aleatoric uncertainty')
    fig.colorbar(im2, ax=axs[1, 0], ticks=[])
    axs[1, 0].axis('off')  # Hide axis

    # Plot epis_1
    im3 = axs[1, 1].imshow(epis_1_np, cmap=cmap, vmin=0, interpolation='none')
    axs[1, 1].set_title('Epistemic uncertainty')
    fig.colorbar(im3, ax=axs[1, 1], ticks=[])
    axs[1, 1].axis('off')  # Hide axis

    # Plot dropout error map mean
    im4 = axs[2, 0].imshow(dropout_error_map_mean_masked, cmap=cmap, vmin=0, interpolation='none')
    axs[2, 0].set_title('Dropout Error [mm]')
    fig.colorbar(im4, ax=axs[2, 0])
    axs[2, 0].axis('off')  # Hide axis

    # Plot evidential error map
    im5 = axs[2, 1].imshow(evidential_error_map_masked, cmap=cmap, vmin=0, interpolation='none')
    axs[2, 1].set_title('Evidential Error [mm]')
    fig.colorbar(im5, ax=axs[2, 1])
    axs[2, 1].axis('off')  # Hide axis

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'plot_{i+1}.png'))
    plt.close(fig)

    print(f'Plot {i+1} saved.')

print('All plots have been generated and saved.')
