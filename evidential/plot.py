import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def tensor_to_array(tensor):
    # Detach tensor
    array = tensor.detach().cpu().clone().numpy()
    return array.squeeze()


def rgb_image(tensor, label):
    image = tensor_to_array(tensor)

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

def line_of_images(all_dict):

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 4, figsize=(10, 4))

    # Call the function for each subplot
    original_image = rgb_image(all_dict["aleatoric"])
    error_map = range_with_bar(all_dict["errormap"])
    aleatoric = range_with_bar(all_dict["aleatoric"])
    epistemic = range_with_bar(all_dict["epistemic"])

    # Add subplots to the main figure
    axs[0].imshow(original_image.get_axes()[0].images[0].get_array(), cmap='viridis')
    axs[1].imshow(aleatoric.get_axes()[0].images[0].get_array(), cmap='viridis')
    axs[2].imshow(epistemic.get_axes()[0].images[0].get_array(), cmap='viridis')

    # Add colorbars
    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(aleatoric.get_axes()[0].images[0], cax=cax1)

    divider2 = make_axes_locatable(axs[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(epistemic.get_axes()[0].images[0], cax=cax2)

    # Set titles for subplots
    axs[2].set_title('aleatoric')
    axs[3].set_title('epistemic')

    # Remove axis labels
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

    return "1"


def evidential(ed):

    aleatoric = ed["aleatoric"]
    aleatoric = aleatoric.detach().cpu().clone().numpy()
    aleatoric = aleatoric.squeeze()
    range_with_bar(aleatoric, "aleatoric")

    epistemic = ed["epistemic"]
    epistemic = epistemic.detach().cpu().clone().numpy()
    epistemic = epistemic.squeeze()
    range_with_bar(epistemic, "epistemic")