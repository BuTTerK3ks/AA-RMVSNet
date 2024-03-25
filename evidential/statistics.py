import torch

def read_tensor_from_pt_file(file_path):
    """
    Reads a PyTorch tensor from a .pt file and returns it.

    Parameters:
    file_path (str): Path to the .pt file containing the tensor.

    Returns:
    torch.Tensor: The tensor loaded from the .pt file.
    """
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)

    return tensor


if __name__ == "__main__"
    file_path = "checkpoints/evidential/results/train/173850.pt"
    tensor = read_tensor_from_pt_file(file_path)
    print(tensor.size())
