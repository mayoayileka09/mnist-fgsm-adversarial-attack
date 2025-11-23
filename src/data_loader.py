def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

import torchvision
import torchvision.transforms as transforms
import torch
# ---------------------------
# 2. Data loading
# ---------------------------
def get_dataloaders(batch_size_train=64, batch_size_test=1000):
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False
    )
    return trainloader, testloader

