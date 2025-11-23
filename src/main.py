from data_loader import get_dataloaders, get_device
from model import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
import attacks


# ---------------------------
#  Training & Evaluation
# ---------------------------
def train(net,
    trainloader,
    device: str,
    epochs: int = 3,
    lr: float = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")


@torch.no_grad()
def test_accuracy(net, loader, device: str) -> float:
    net.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    return acc


def main():
    device = get_device()
    print("Using device:", device)

    # 1) Data
    trainloader, testloader = get_dataloaders()

    # 2) Model
    net = LeNet().to(device)

    # 3) Train baseline model
    print("\n=== Training baseline model ===")
    train(net, trainloader, device, epochs=3, lr=0.001)

    # 4) Evaluate clean accuracy
    print("\n=== Evaluating on clean test set ===")
    clean_acc = test_accuracy(net, testloader, device)
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")

    # 5) PGD attacks for different epsilons
    epsilons = [0.05, 0.1, 0.2, 0.3]
    alpha = 0.01       # step size for each PGD iteration
    num_iters = 40     # number of PGD steps

    print("\n=== PGD Attack Success Rates ===")
    print("epsilon\tASR (%)")
    for eps in epsilons:
        asr = attacks.attack_success_rate(net, testloader, eps, alpha, num_iters, device)
        print(f"{eps:.2f}\t{asr * 100:.2f}")

    # 6) Show a visual example for one epsilon
    print("\nShowing PGD adversarial example for epsilon = 0.1")
    attacks.show_adversarial_example(net, testloader, epsilon=0.05, alpha=alpha, num_iters=num_iters, device=device)
    attacks.show_adversarial_example(net, testloader, epsilon=0.1, alpha=alpha, num_iters=num_iters, device=device)
    attacks.show_adversarial_example(net, testloader, epsilon=0.2, alpha=alpha, num_iters=num_iters, device=device)
    attacks.show_adversarial_example(net, testloader, epsilon=0.3, alpha=alpha, num_iters=num_iters, device=device)


if __name__ == "__main__":
    main()