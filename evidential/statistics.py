import torch

def std_prob(prob_volume):

    # Schritt 1: Finden Sie die Indizes der maximalen Werte in der Dimension 1
    max_indices = torch.argmax(prob_volume, dim=1)

    # Schritt 2: Extrahieren Sie die maximalen Werte
    # Dazu müssen Sie eine geeignete Indexierungsmethode verwenden, z.B. torch.gather
    max_values = torch.gather(prob_volume, 1, max_indices.unsqueeze(1))

    # Schritt 3: Berechnen Sie die Standardabweichung um diese maximalen Werte
    # Zentrieren Sie den Tensor um die maximalen Werte
    centered_tensor = prob_volume - max_values

    # Berechnen Sie die Standardabweichung entlang der Dimension 1
    std_dev = torch.std(centered_tensor, dim=1)

    return std_dev

def divide_alea_epis(evidential_outputs):
    division = evidential_outputs["aleatoric"]/evidential_outputs["epistemic"]
    return division