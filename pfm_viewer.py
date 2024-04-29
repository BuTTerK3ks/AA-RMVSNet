import re
import cv2
import numpy as np
from PIL import Image


def readPF(filename):
    """Read named PF file into Numpy array"""

    # Slurp entire file into memory as binary 'bytes'
    with open(filename, 'rb') as f:
        data = f.read()

    # Check correct header, return None if incorrect
    if not re.match(b'Typ=Pic98::TPlane<float>', data):
        return None

    # Get Lines and Columns, both must be present, else return None
    L = re.search(b'Lines=(\d+)', data)
    C = re.search(b'Columns=(\d+)', data)
    if not (L and C):
        return None
    height = int(L.groups()[0])
    width = int(C.groups()[0])
    print(f'DEBUG: Height={height}, width={width}')

    # Take the data from the END of the file in case other header lines added at start
    na = np.frombuffer(data[-4 * height * width:], dtype=np.dtype('<f4')).reshape((height, width))

    # Some debug stuff
    min, max, mean = na.min(), na.max(), na.mean()
    print(f'DEBUG: min={min}, max={max}, mean={mean}')

    return na


################################################################################
# Main
################################################################################
na = readPF('PF file.PF')

################################################################################
# Use either of the following to save the image:
################################################################################
# Save with OpenCV as scaled PNG
u16 = (65535 * (na - np.min(na)) / np.ptp(na)).astype(np.uint16)
cv2.imwrite('OpenCV.png', u16)

# Convert to PIL Image and save as TIFF
pi = Image.fromarray(na, mode='F')
pi.save('PIL.tif')