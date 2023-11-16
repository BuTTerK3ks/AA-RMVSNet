import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def image_with_bar(array, type):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the colored picture
    im = ax.imshow(array, cmap='viridis')  # You can change 'viridis' to another colormap

    # Add a colorbar legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    if type == "aleatoric":
        cbar.set_label('Aleatoric uncertainty')


    # Show the plot
    plt.show()


def evidential(ed):

    aleatoric = ed["aleatoric"]
    aleatoric = aleatoric.detach().cpu().clone().numpy()
    aleatoric = aleatoric.squeeze()
    image_with_bar(aleatoric, "aleatoric")

    epistemic = ed["epistemic"]
    epistemic = epistemic.detach().cpu().clone().numpy()
    epistemic = epistemic.squeeze()
    image_with_bar(epistemic, "epistemic")