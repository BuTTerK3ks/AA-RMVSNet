import torch
import torch.nn as nn
from torchviz import make_dot


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.gn = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class IntraViewAAModule(nn.Module):
    def __init__(self):
        super(IntraViewAAModule, self).__init__()
        # Define the layers for the IntraViewAAModule here
        self.dummy_layer = nn.Identity()

    def forward(self, x0, x1, x2):
        # Resize x1 and x2 to match the dimensions of x0
        x1 = nn.functional.interpolate(x1, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x2 = nn.functional.interpolate(x2, size=x0.shape[2:], mode='bilinear', align_corners=False)
        return self.dummy_layer(x0 + x1 + x2)  # Example implementation


class CustomModel(nn.Module):
    def __init__(self, base_filter):
        super(CustomModel, self).__init__()
        self.init_conv = nn.Sequential(
            ConvGnReLU(3, base_filter, kernel_size=3, stride=1, dilation=1),
            ConvGnReLU(base_filter, base_filter * 2, kernel_size=3, stride=1, dilation=1)
        )
        self.conv0 = ConvGnReLU(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
        self.conv1 = ConvGnReLU(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, dilation=1)
        self.conv2 = ConvGnReLU(base_filter * 4, base_filter * 4, kernel_size=3, stride=2, dilation=1)
        self.intraAA = IntraViewAAModule()

    def forward(self, x):
        x = self.init_conv(x)
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        return self.intraAA(x0, x1, x2)


# Instantiate the model
model = CustomModel(base_filter=64)

# Create a dummy input tensor with appropriate size
x = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image size

# Pass the input through the model to generate the graph
y = model(x)

# Generate and save the visualization
make_dot(y, params=dict(list(model.named_parameters()))).render("custom_model_torchviz", format="png")
