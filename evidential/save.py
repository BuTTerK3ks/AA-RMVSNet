from evidential.plot import *

save_path = "outputs_dtu/evidential"

def save_evidential(image_outputs, evidential_outputs):
    onedict = {**image_outputs, **evidential_outputs}
    returned_figure = line_of_images(onedict)

    raise NotImplementedError