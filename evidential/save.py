from evidential.plot import *
from matplotlib import pyplot as plt
import datetime
import os


current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

save_path = "outputs_dtu/evidential/"
filename = f"{formatted_datetime}.png"

def save_grid(logger, image_outputs, evidential_outputs):
    onedict = {**image_outputs, **evidential_outputs}
    returned_figure = grid_of_images(onedict)
    #plt.savefig(save_path + filename)
    plt.close('all')

    def save_images(logger, mode, images_dict, global_step):
        images_dict = tensor2numpy(images_dict)

        def preprocess(name, img):
            if not (len(img.shape) == 3 or len(img.shape) == 4):
                raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
            if len(img.shape) == 3:
                img = img[:, np.newaxis, :, :]
            img = torch.from_numpy(img[:1])
            return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

        for key, value in images_dict.items():
            if not isinstance(value, (list, tuple)):
                name = '{}/{}'.format(mode, key)
                logger.add_image(name, preprocess(name, value), global_step)
            else:
                for idx in range(len(value)):
                    name = '{}/{}_{}'.format(mode, key, idx)
                    logger.add_image(name, preprocess(name, value[idx]), global_step)

def save_pytorch(directory, mode, global_step, image_outputs, evidential_outputs):
    total_dict = {**image_outputs, **evidential_outputs}
    del total_dict['aleatoric_1_by_total']
    del total_dict['aleatoric_2_by_total']
    del total_dict['epistemic_1_by_total']
    del total_dict['epistemic_2_by_total']
    del total_dict['alea_1']
    del total_dict['alea_2']
    del total_dict['epis_1']
    del total_dict['epis_2']
    path_to_store = directory + "/results/" + str(mode) + "/"

    # Check if the directory exists, if not, create it
    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store)

    filename = str(global_step)
    torch.save(total_dict, path_to_store + filename + '.pt')
