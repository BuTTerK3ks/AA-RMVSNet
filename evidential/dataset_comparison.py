import torch
import torch.nn as nn
import torch.nn.functional as F
from graphviz import Digraph


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.gn = nn.GroupNorm(32, out_channels)  # Assuming 32 groups for GroupNorm
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        return self.relu(x)


class IntraViewAAModule(nn.Module):
    def forward(self, x0, x1, x2):
        # Adjust tensor sizes to match for summation
        x1 = F.interpolate(x1, size=x0.shape[2:])
        x2 = F.interpolate(x2, size=x0.shape[2:])
        return x0 + x1 + x2


class Model(nn.Module):
    def __init__(self, base_filter):
        super(Model, self).__init__()
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


def make_dot(var):
    """ Produces Graphviz representation of PyTorch autograd graph """
    if isinstance(var, torch.autograd.Variable):
        var = var.grad_fn
    dot = Digraph()
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(fn):
        if fn not in seen:
            if torch.is_tensor(fn):
                dot.node(str(id(fn)), size_to_str(fn.size()), shape='oval')
            elif hasattr(fn, 'variable'):
                u = fn.variable
                dot.node(str(id(fn)), size_to_str(u.size()), shape='oval')
            else:
                dot.node(str(id(fn)), str(type(fn).__name__))
            seen.add(fn)
            if hasattr(fn, 'next_functions'):
                for u in fn.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(fn)))
                        add_nodes(u[0])
            if hasattr(fn, 'saved_tensors'):
                for t in fn.saved_tensors:
                    dot.edge(str(id(t)), str(id(fn)))
                    add_nodes(t)

    add_nodes(var)
    return dot


# Create a model instance and visualize it
model = Model(base_filter=64)
x = torch.randn(1, 3, 224, 224)  # Example input
y = model(x)

dot = make_dot(y)
file_path = "model_visualization"
dot.render(file_path, format='png')

print(f"Model visualization saved to {file_path}.png")
