from evidential.plot import *
from matplotlib import pyplot as plt
import datetime


current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

save_path = "outputs_dtu/evidential/"
filename = f"{formatted_datetime}.png"

def save_evidential(image_outputs, evidential_outputs):
    onedict = {**image_outputs, **evidential_outputs}
    returned_figure = grid_of_images(onedict)
    plt.savefig(save_path + filename)
    plt.close('all')

def draw_disparity(outputs):
    plt.clf()


    outputs = outputs
    probabilities = outputs['probability_volume']
    pixel_values = probabilities[0, :, 64, 80].detach().cpu()

    # Plotting the values
    #plt.figure(figsize=(10, 6))
    plt.plot(pixel_values, color='blue')
    plt.title('Disparity diagramm')
    plt.xlabel('Probability')
    plt.ylabel('Depth')
    plt.show()

    plt.close('all')