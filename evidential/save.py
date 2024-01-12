from evidential.plot import *
from matplotlib import pyplot as plt
import datetime


current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

save_path = "outputs_dtu/evidential/"
filename = f"{formatted_datetime}.png"

def save_errormap(image_outputs, evidential_outputs):
    onedict = {**image_outputs, **evidential_outputs}
    returned_figure = grid_of_images(onedict)
    #plt.savefig(save_path + filename)
    plt.close('all')
