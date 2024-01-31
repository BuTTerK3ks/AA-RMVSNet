import torch

def std_prob(probabilities):

    # Berechnen Sie die Standardabweichung entlang der Dimension 1
    std_dev = torch.std(probabilities, dim=1)

    return std_dev

def divide_alea_epis(evidential_outputs):
    division = evidential_outputs["aleatoric"]/evidential_outputs["epistemic"]
    return division