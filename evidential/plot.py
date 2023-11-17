import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def tensor_to_array(tensor):
    # Detach tensor
    array = tensor.detach().cpu().clone().numpy()
    return array.squeeze()


def rgb_image(tensor):
    image = tensor_to_array(tensor)

    # Transpose the image to have channels as the last dimension
    image = np.transpose(image, (1, 2, 0))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the RGB image
    ax.imshow(image)

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])


    return fig

def range_with_bar(tensor):
    array = tensor_to_array(tensor)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the colored picture
    im = ax.imshow(array, cmap='viridis')  # You can change 'viridis' to another colormap

    return fig

def grid_of_images(all_dict):

    # Create a figure with three subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Call the function for each subplot
    original_image = tensor_to_array(all_dict["ref_img"])
    # Transpose the image to have channels as the last dimension
    original_image = np.transpose(original_image, (1, 2, 0))
    error_map = tensor_to_array(all_dict["errormap"])
    aleatoric = tensor_to_array(all_dict["aleatoric"])
    epistemic = tensor_to_array(all_dict["epistemic"])

    # Add subplots to the main figure
    im1 = axs[0, 0].imshow(original_image, cmap='seismic')
    im2 = axs[0, 1].imshow(error_map, cmap='viridis')
    im3 = axs[1, 0].imshow(aleatoric, cmap='viridis')
    im4 = axs[1, 1].imshow(epistemic, cmap='viridis')

    # Add colorbars
    divider1 = make_axes_locatable(axs[1, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax1)

    divider2 = make_axes_locatable(axs[1, 1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, cax=cax2)

    divider3 = make_axes_locatable(axs[0, 1])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax3)

    # Set titles for subplots
    axs[0, 0].set_title('Image')
    axs[0, 1].set_title('Error')
    axs[1, 0].set_title('Aleatoric')
    axs[1, 1].set_title('Epistemic')

    # Remove axis labels
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()

    return fig
