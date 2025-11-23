import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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