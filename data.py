import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_mnist_batch(size, flatten=False):
    torch.manual_seed(42)

    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=size,
                                              shuffle=True)

    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    return images, labels



if __name__ == "__main__":
    images, labels = get_mnist_batch(10)

    print(images.shape)
    print(labels.shape)
