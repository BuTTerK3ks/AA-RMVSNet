import torch

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    # Print CUDA device information
    print(f"CUDA is available! Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Create a tensor and move it to GPU
    x = torch.rand(3, 3)
    x = x.cuda()
    print(f"Random CUDA tensor:\n{x}")

if __name__ == "__main__":
    main()
