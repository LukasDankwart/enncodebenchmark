import torch
import torch.nn as nn
import onnx
import numpy as np
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import config


class AdversarialExample1(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        if hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())

            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        else:
            layers.append(nn.Linear(input_dim, output_dim))

        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def export_onnx(self, path):
        self.eval()

        input_dim = [1] + [self.input_dim]
        rand_input = torch.randn(input_dim)
        input_names = ["input"]
        output_names = ["output"]

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        torch.onnx.export(
            self,
            rand_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"ONNX model was exported successfully to given path: {path}")


class AdversarialExample2(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        c, h, w = input_dim
        h = h - 2
        w = w - 2
        h = h - 2
        w = w - 2

        self.flatten_size = 8 * h * w

        self.model = nn.Sequential(
            nn.Conv2d(c, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

    def export_onnx(self, path):
        self.eval()

        rand_input = torch.randn(1, *self.input_dim)
        input_names = ["input"]
        output_names = ["output"]

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        torch.onnx.export(
            self,
            rand_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"ONNX model was exported successfully to given path: {path}")


class Net(nn.Module):
    # define layers of neural network
    def __init__(self, hidden_size=50):
        super().__init__()
        self.hidden1 = nn.Linear(784, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()

    # define forward pass of neural network
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

    def export_onnx(self, path):
        self.eval()

        rand_input = torch.randn(1, 784)
        input_names = ["input"]
        output_names = ["output"]

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        torch.onnx.export(
            self,
            rand_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"ONNX model was exported successfully to given path: {path}")


class Net2(nn.Module):
    # define layers of neural network
    def __init__(self, hidden_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, (4, 4), (2, 2), 0)
        self.conv2 = nn.Conv2d(2, 2, (4, 4), (2, 2), 0)
        self.hidden1 = nn.Linear(5 * 5 * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()

    # define forward pass of neural network
    def forward(self, x):
        self.x1 = self.conv1(x)
        self.x2 = self.relu(self.x1)
        self.x3 = self.conv2(self.x2)
        self.x4 = self.relu(self.x3)
        self.x5 = self.hidden1(self.x4.view((-1, 5 * 5 * 2)))
        self.x6 = self.relu(self.x5)
        self.x7 = self.output(self.x6)
        return self.relu(self.x7)

    def export_onnx(self, path):
        self.eval()

        rand_input = torch.randn(1, 1, 28, 28)
        input_names = ["input"]
        output_names = ["output"]

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        torch.onnx.export(
            self,
            rand_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"ONNX model was exported successfully to given path: {path}")


class NoSoftmaxNet(nn.Module):
    # define layers of neural network
    def __init__(self, hidden_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, (4, 4), (2, 2), 0)
        self.conv2 = nn.Conv2d(2, 2, (4, 4), (2, 2), 0)
        self.fc = nn.Linear(2 * 5 * 5, 10)
        self.relu = nn.ReLU()

    # define forward pass of neural network
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.fc(x.view(-1, 5 * 5 * 2))
        x = self.relu(x)
        return x


def export_onnx(model, path, flatten=False):
    model.eval()

    rand_input = torch.rand(1, 784) if flatten else torch.randn(1, 1, 28, 28)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        rand_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training runs on device: {device}")


    """
    # Code for Training FC MNIST
    fc_mnist = AdversarialExample1(784, 10, [64])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST Mean/Std
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 1000}

    dataset1 = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    optimizer = optim.Adadelta(fc_mnist.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(20):
        train(fc_mnist, device, train_loader, optimizer, epoch)
        evaluate(fc_mnist, device, test_loader)
        scheduler.step()

    path = "models/mnist/fc_mnist.onnx"
    tmp_path = "models/mnist/fc_mnist_with_bounds.onnx"

    export_onnx(fc_mnist, path, flatten=True)
    model_proto = onnx.load(path)
    graph = model_proto.graph
    input_names = [node.name for node in graph.input]
    initializer_names = {x.name for x in graph.initializer}
    real_inputs = [name for name in input_names if name not in initializer_names]

    input_nodes = [node for node in model_proto.graph.input if node.name in real_inputs]
    input_tensor = input_nodes[0]
    input_shape = [max(dim.dim_value, 1) for dim in input_tensor.type.tensor_type.shape.dim]
    n_inputs = input_shape[-1]
    lb = np.maximum(0, 0)
    ub = np.minimum(1, 1)
    input_bounds = {}
    for i in range(784):
            input_bounds[(0, i)] = (config.MNIST_LB, config.MNIST_LB)
    write_onnx_model_with_bounds(tmp_path, None, input_bounds)

    """
    # Code for training conv net
    conv_net = NoSoftmaxNet(hidden_size=10).to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST Mean/Std
    ])

    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 1000}

    dataset1 = datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    optimizer = optim.Adadelta(conv_net.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(20):
        train(conv_net, device, train_loader, optimizer, epoch)
        evaluate(conv_net, device, test_loader)
        scheduler.step()

    path = "models/mnist/conv_net.onnx"
    tmp_path = "models/mnist/conv_net_with_bounds.onnx"

    export_onnx(conv_net, path)
    model_proto = onnx.load(path)
    graph = model_proto.graph
    input_names = [node.name for node in graph.input]
    initializer_names = {x.name for x in graph.initializer}
    real_inputs = [name for name in input_names if name not in initializer_names]

    input_nodes = [node for node in model_proto.graph.input if node.name in real_inputs]
    input_tensor = input_nodes[0]
    input_shape = [max(dim.dim_value, 1) for dim in input_tensor.type.tensor_type.shape.dim]
    n_inputs = input_shape[-1]
    lb = np.maximum(0, 0)
    ub = np.minimum(1, 1)
    input_bounds = {}
    for i in range(28):
        for j in range(28):
            input_bounds[(0, i, j)] = (float(0.0), float(1.0))
    write_onnx_model_with_bounds(tmp_path, None, input_bounds)