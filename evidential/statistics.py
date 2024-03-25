import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

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
    std_devs = data_dict['std_dev'].detach().cpu().numpy().flatten()
    mask = data_dict['mask'].detach().cpu().numpy().flatten().astype(bool)

    # Apply the mask to filter the data
    filtered_errors = errors[mask]
    filtered_std_devs = std_devs[mask]

    if len(filtered_errors) == 0 or len(filtered_std_devs) == 0:
        print("No data points remain after filtering.")
        return

    # Compute optimal bin edges for both errors and standard deviations using the 'auto' strategy
    error_bins = np.histogram_bin_edges(filtered_errors, bins='auto')
    std_dev_bins = np.histogram_bin_edges(filtered_std_devs, bins='auto')

    # Create the heatmap
    plt.hist2d(filtered_errors, filtered_std_devs, bins=[error_bins, std_dev_bins], cmap=plt.cm.jet, density=True)
    plt.colorbar()

    t5 = max(filtered_errors)

    # Explicitly set x-axis limits to the range of your filtered error data
    plt.xlim(min(filtered_errors), 5)

    plt.xlabel('Error per Pixel (cm)')
    plt.ylabel('Standard Deviation (cm)')
    plt.title('Heatmap of Filtered Error vs. Standard Deviation')

    plt.show()

def create_pixelwise_heatmap(data_dict):
    """
    Creates a pixel-wise heatmap from a 2D array of errors, displaying each pixel in its original location,
    and shows a colorbar without scale (numbers).

    Parameters:
    data_dict (dict): A dictionary containing a 2D or 3D tensor/array 'error_map'.
                      The tensor/array should be 2D, or 3D with the first dimension being of size 1.
    """
    # Assuming 'error_map' might be 3D with the first dimension of size 1
    errors = data_dict['error_map'].detach().cpu().numpy()

    # Handle the case where errors might have an extra first dimension of size 1
    if errors.ndim == 3 and errors.shape[0] == 1:
        errors = errors[0]  # Select the first element of the first dimension to make it 2D

    # Display the heatmap
    img = plt.imshow(errors, cmap=plt.cm.jet)  # 'origin' parameter can be added if needed
    plt.title('Error/Std Deviation')

    # Create a colorbar without scale (numbers)
    cbar = plt.colorbar(img)
    cbar.set_ticks([])  # Removes the ticks

    plt.axis('off')  # Optional: Remove the axis if you don't want to show x and y ticks and labels
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
    Reads a dictionary of tensors from a .pt file and returns it.

    Parameters:
    file_path (str): Path to the .pt file containing the dictionary of tensors.

    Returns:
    dict: A dictionary where keys are the original keys in the file, and values are the tensors.
    """
    # Load the dictionary from the .pt file
    tensor_dict = torch.load(file_path)

    # Optionally, you can add checks here to ensure that the loaded object is a dictionary
    # and each value in the dictionary is indeed a tensor.

    return tensor_dict

if __name__ == "__main__":
    file_path = "/home/grannemann/PycharmProjects/AA-RMVSNet/checkpoints/evidential/results/train/173850.pt"
    results = read_tensors_from_pt_file(file_path)
    create_filtered_heatmap(results)
    create_pixelwise_heatmap(results)
    show_ref_image(results)

    print("EoS")