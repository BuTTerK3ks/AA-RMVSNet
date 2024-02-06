import torch
from evidential.models import *
from evidential.save import *
from tensorboardX import SummaryWriter
from torchsummary import summary
from models import *
from torchviz import make_dot



def std_prob(probabilities):

    # Berechnen Sie die Standardabweichung entlang der Dimension 1
    std_dev = torch.std(probabilities, dim=1)

    return std_dev

def divide_alea_epis(evidential_outputs):
    division = evidential_outputs["aleatoric"]/evidential_outputs["epistemic"]
    return division

def model_structure():
    dummy_aarmvsnet_model = AARMVSNetWrapper().cuda()
    summary(dummy_aarmvsnet_model, input_size=(5, 3, 128, 160))

    dummy_evidential_model = EvidentialWrapper().cuda()
    summary(dummy_evidential_model, input_size=(32, 128, 160))
    torch.cuda.empty_cache()  # Clear the CUDA cache

if __name__ == "__main__":

    dummy_evidential_model = EvidentialWrapper().cuda()
    dummy_aarmvsnet_model = AARMVSNetWrapper().cuda()


    dummy_input = torch.randn(1, 5, 3, 128, 160).cuda()
    output = dummy_aarmvsnet_model(dummy_input)
    output = output["probability_volume"]
    dot = make_dot(output.mean(), params=dict(dummy_evidential_model.named_parameters()))
    dot.render("/home/grannemann/Downloads/dummy_aarmvsnet_model", format="png")


    dummy_input = torch.randn(1, 32, 128, 160).cuda()
    output = dummy_evidential_model(dummy_input)
    dot = make_dot(output, params=dict(dummy_evidential_model.named_parameters()))
    dot.render("/home/grannemann/Downloads/dummy_evidential_model", format="png")

