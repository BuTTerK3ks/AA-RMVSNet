import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def save_png(array, filepath, title="", mode="relative"):
    """
    Exports a depth map from a given 2D numpy array to a PNG file using the specified colormap.
    Normalizes the array to the [0, 1] range, handles None values by setting them to black, and filters values over 1000.

    Parameters:
        array (np.ndarray): A 2D numpy array representing the depth map.
        filepath (str): The path to save the PNG image file.
        colormap (str): The name of the colormap to use.
    """
    colormap = plt.cm.jet

    # Check for None, which is not natively supported in numpy arrays; assume input might be object type
    if array.dtype == np.object:
        array = np.where(array == None, 0, array).astype(float)  # Replace None with 0 and ensure type is float

    # Ensure the array is a 2D numpy array
    if array.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    if mode == "depth":
        # Invert the array values for depth mode
        valid_mask = ~np.isnan(array)  # Mask to ignore NaN values in computation
        min_val = array[valid_mask].min()
        max_val = array[valid_mask].max()
        array = (max_val - array) + min_val  # Invert the values and add min_val to avoid starting from 0
        array = array * 10  # Scale the values by 10
        array[~valid_mask] = 0  # Set NaN values to 0 (black)

    elif mode == "relative":
        # Normalize the array to [0, 1]
        array = array.astype(np.float32)  # Ensure array is float for processing
        valid_mask = ~np.isnan(array)  # Mask to ignore NaN values in computation
        min_val = array[valid_mask].min()
        max_val = array[valid_mask].max()
        array = (array - min_val) / (max_val - min_val)
        array[~valid_mask] = 0  # Set NaN values to 0 (black)


    # Create the plot with the specified colormap and no axes for clarity
    img = plt.imshow(array, cmap=colormap)
    plt.axis('off')  # Turn off the axis
    plt.title(title)

    # Create a colorbar without scale (numbers)
    cbar = plt.colorbar(img)
    if mode == "relative":
        cbar.set_ticks([])  # Removes the ticks

    # Save the image to the specified filepath
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free up memory
