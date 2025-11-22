
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# This script trains a LeNet-like CNN on MNIST and evaluates robustness using
# a multi-step PGD (Projected Gradient Descent) adversarial attack.
# PGD iteratively applies small FGSM-like steps and projects back into an
# epsilon-ball around the original image, providing a stronger attack.



# ---------------------------
# 1. Device helper
# ---------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


# ---------------------------
# 3. LeNet model
# ---------------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST is 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # -> 6 x 24 x 24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # -> 16 x 8 x 8 (after pooling)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 6 x 12 x 12

        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 16 x 4 x 4

        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------
# 4. Training & Evaluation
# ---------------------------
def train(model, trainloader, device, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    return acc


#----------------------------
# 5. PGD attack
# ---------------------------
def pgd_attack(model, images, labels, epsilon, alpha, num_iters, device):
    """
    Multi-step PGD (Projected Gradient Descent) attack.

    Starts from the clean image and iteratively performs:
        x_{t+1} = Π_{B_eps(x0)} ( x_t + alpha * sign(∇_x L(model(x_t), y)) )

    where Π_{B_eps(x0)} projects back into the L-infinity epsilon-ball
    around the original image x0 and clips to the valid pixel range [0, 1].

    This is a stronger attack than single-step FGSM because it takes
    multiple smaller steps instead of one big step.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Start from the original images (could also add small random noise here)
    adv_images = images.clone().detach()

    model.eval()
    for _ in range(num_iters):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = adv_images.grad.sign()
            adv_images = adv_images + alpha * grad_sign

            # Project perturbation back into the epsilon L-infinity ball
            delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(images + delta, 0, 1)

        adv_images = adv_images.detach()

    return adv_images


def attack_success_rate(model, loader, epsilon, alpha, num_iters, device):
    """
    ASR = fraction of perturbed samples that are misclassified under PGD.
    """
    model.eval()
    total = 0
    fooled = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_iters, device)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        fooled += (predicted != labels).sum().item()

    return fooled / total


# ---------------------------
# 6. Visualization
# ---------------------------
def show_adversarial_example(model, loader, epsilon, alpha, num_iters, device):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    adv_images = pgd_attack(model, images, labels, epsilon, alpha, num_iters, device)

    idx = 0  # just show the first example
    clean_img = images[idx].cpu().squeeze()
    adv_img = adv_images[idx].cpu().squeeze()

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.title("Clean")
    plt.imshow(clean_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Adversarial (ε={epsilon})")
    plt.imshow(adv_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------
# 7. Main script
# ---------------------------
def main():
    device = get_device()
    print("Using device:", device)

    # 1) Data
    trainloader, testloader = get_dataloaders()

    # 2) Model
    model = LeNet().to(device)

    # 3) Train baseline model
    print("\n=== Training baseline model ===")
    train(model, trainloader, device, epochs=3, lr=0.001)

    # 4) Evaluate clean accuracy
    print("\n=== Evaluating on clean test set ===")
    clean_acc = test_accuracy(model, testloader, device)
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")

    # 5) PGD attacks for different epsilons
    epsilons = [0.05, 0.1, 0.2]
    alpha = 0.01       # step size for each PGD iteration
    num_iters = 40     # number of PGD steps

    print("\n=== PGD Attack Success Rates ===")
    print("epsilon\tASR (%)")
    for eps in epsilons:
        asr = attack_success_rate(model, testloader, eps, alpha, num_iters, device)
        print(f"{eps:.2f}\t{asr * 100:.2f}")

    # 6) Show a visual example for one epsilon
    print("\nShowing PGD adversarial example for epsilon = 0.1")
    show_adversarial_example(model, testloader, epsilon=0.1, alpha=alpha, num_iters=num_iters, device=device)


if __name__ == "__main__":
    main()