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
def show_adversarial_example(model, loader, epsilons, alpha, num_iters, device, samples=None):
    """
    Show clean images and multiple PGD-adversarial versions
    for different epsilons in a single figure.

    epsilons: iterable of epsilon values (e.g., [0.05, 0.1, 0.2, 0.3])
    samples: list of indices into the batch (e.g., [0, 1, 2, 3])
    """
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    # Default: first 4 samples from the batch if none provided
    if samples is None or len(samples) == 0:
        max_samples = min(4, images.size(0))
        samples = list(range(max_samples))

    num_rows = len(samples)
    num_cols = 1 + len(epsilons)  # clean + one col per epsilon

    plt.figure(figsize=(4 * num_cols, 3 * num_rows))

    # Precompute adversarial batches for each epsilon (more efficient)
    adv_batches = {}
    for eps in epsilons:
        adv_batches[eps] = pgd_attack(
            model, images, labels, eps, alpha, num_iters, device
        )

    # For each sample (row)
    for row_idx, sample_idx in enumerate(samples):
        clean_img = images[sample_idx].cpu().squeeze()

        # Column 1: clean image
        plt.subplot(num_rows, num_cols, row_idx * num_cols + 1)
        if row_idx == 0:
            plt.title("Clean")
        plt.imshow(clean_img, cmap="gray")
        plt.axis("off")

        # Columns 2..: adversarial for each epsilon
        for col_offset, eps in enumerate(epsilons, start=1):
            adv_img = adv_batches[eps][sample_idx].cpu().squeeze()

            plt.subplot(num_rows, num_cols, row_idx * num_cols + 1 + col_offset)
            if row_idx == 0:
                plt.title(f"PGD ε={eps}")
            plt.imshow(adv_img, cmap="gray")
            plt.axis("off")

    plt.tight_layout()
    plt.show()