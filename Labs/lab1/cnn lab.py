# ========== Part A ==========
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Transforms: PIL image -> tensor, then channel-wise normalisation
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,255] -> [0.0, 1.0]
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=False, num_workers=2)

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")
print(f"Image shape: {train_set[0][0].shape}")  # torch.Size([3, 32, 32])

# ========== Part B ==========
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    """
    Input:  3 * 32 * 32 = 3,072 values (flattened)
    Hidden: 512 -> 256 neurons
    Output: 10 classes
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (B,3,32,32) -> (B,3072)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

dense_model = DenseNet()
dense_params = sum(p.numel() for p in dense_model.parameters())
print(f"DenseNet total parameters: {dense_params:,}")

import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, predicted = model(images).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct / total
        history['train_loss'].append(epoch_loss)
        history['test_acc'].append(epoch_acc)
        print(f"Epoch {epoch:2d}/{epochs} "
              f"Loss: {epoch_loss:.4f} "
              f"Test Acc: {epoch_acc:.1f}% "
              f"({time.time() - t0:.1f}s)")

    return history

print("=" * 50)
print("Training Dense Network")
print("=" * 50)
dense_history = train_model(dense_model, train_loader,
                            test_loader, epochs=10)

# ========== Part C ==========
class SimpleCNN(nn.Module):
    """
    3 convolutional blocks + classification head.
    Block 1: Conv (32) x2 -> MaxPool -> 16x16x32
    Block 2: Conv (64) x2 -> MaxPool -> 8x8x64
    Block 3: Conv (128) -> MaxPool -> 4x4x128
    Head:    GlobalAvgPool -> Dense (256) -> Dense (10)
    """
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        # PART E CHALLENGE
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Avg Pool -> (B,128,1,1)
            nn.Flatten(),             # -> (B,128)
            # nn.Linear(128, 256),  # ORIGINAL ARCITECTURE
            nn.Linear(256, 256),    # PART E CHALLENGE
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # PART E CHALLENGE
        x = self.block4(x)
        return self.classifier(x)

cnn_model = SimpleCNN()
cnn_params = sum(p.numel() for p in cnn_model.parameters())
print(f"CNN total parameters: {cnn_params:,}")
print(f"Dense parameters: {dense_params:,}")
print(f"Ratio: {dense_params / cnn_params:.1f}x more in Dense")

print("=" * 50)
print("Training CNN")
print("=" * 50)
cnn_history = train_model(cnn_model, train_loader,
                          test_loader, epochs=15, lr=1e-3)

# ========== Part D ==========
def plot_kernels(model, n_filters=32):
    first_conv = list(model.block1.children())[0]  # nn.Conv2d
    kernels = first_conv.weight.data.cpu().numpy()  # (32, 3, 3, 3)

    cols = 8
    rows = n_filters // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.8))

    for idx, ax in enumerate(axes.flatten()):
        if idx < n_filters:
            k = np.transpose(kernels[idx], (1, 2, 0))  # HWC
            k = (k - k.min()) / (k.max() - k.min() + 1e-8)
            ax.imshow(k)
            ax.set_title(f'K{idx + 1}', fontsize=7)
            ax.axis('off')

    plt.suptitle("Learned First-Layer Kernels (3x3 RGB)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("kernels.png", dpi=120)
    plt.show()
    print("Saved kernels.png")

plot_kernels(cnn_model)


def get_feature_maps(model, dataset, img_idx=7):
    model.eval()
    img, label = dataset[img_idx]

    # Show original image (undo normalisation for display)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    disp = np.clip(std * img.numpy().transpose(1, 2, 0) + mean, 0, 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(disp)
    plt.title(f"Input: {CLASSES[label]}")
    plt.axis('off')
    plt.savefig("input_image.png", dpi=120)
    plt.show()

    # Register hooks to capture block outputs
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    h1 = model.block1.register_forward_hook(make_hook('block1'))
    h2 = model.block2.register_forward_hook(make_hook('block2'))
    h3 = model.block3.register_forward_hook(make_hook('block3'))

    with torch.no_grad():
        _ = model(img.unsqueeze(0).to(DEVICE))

    h1.remove(); h2.remove(); h3.remove()
    return activations, CLASSES[label]


def plot_feature_maps(activations, block_name, n_maps=32):
    fmaps = activations[block_name][0].numpy()  # (C, H, W)
    n_maps = min(n_maps, fmaps.shape[0])
    cols = 8
    rows = n_maps // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.8))

    for idx, ax in enumerate(axes.flatten()):
        if idx < n_maps:
            ax.imshow(fmaps[idx], cmap='viridis')
            ax.set_title(f'F{idx + 1}', fontsize=7)
            ax.axis('off')

    h, w = fmaps.shape[1], fmaps.shape[2]
    plt.suptitle(
        f"Feature Maps -- {block_name} "
        f"({n_maps} filters, {h}x{w})",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"fmaps_{block_name}.png", dpi=120)
    plt.show()
    print(f"Saved fmaps_{block_name}.png")


# Run
activations, true_class = get_feature_maps(cnn_model, test_set, img_idx=7)

print("\nFeature map shapes:")
for name, act in activations.items():
    print(f"  {name}: {tuple(act.shape[1:])}")

plot_feature_maps(activations, 'block1')
plot_feature_maps(activations, 'block2')
plot_feature_maps(activations, 'block3')

# ========== Part E ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Training Loss
ax = axes[0]
ax.plot(dense_history['train_loss'], 'o-', color='#E53935',
        linewidth=2, label='Dense Network')
ax.plot(cnn_history['train_loss'], 's-', color='#1A237E',
        linewidth=2, label='CNN')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Training Loss', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(dense_history['test_acc'], 'o-', color='#E53935', linewidth=2,
        label=f"Dense (peak: {max(dense_history['test_acc']):.1f}%)")
ax.plot(cnn_history['test_acc'], 's-', color='#1A237E', linewidth=2,
        label=f"CNN (peak: {max(cnn_history['test_acc']):.1f}%)")
ax.axhline(10, color='gray', linestyle='--', alpha=0.5,
           label='Random baseline')
ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Dense Network vs CNN -- CIFAR-10',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()

# Summary table
print("\n" + "=" * 55)
print(f"{'Model':<15} {'Parameters':>12} {'Peak Test Acc':>15}")
print("-" * 55)
print(f"{'Dense Network':<15} {dense_params:>12,} "
      f"{max(dense_history['test_acc']):>14.1f}%")
print(f"{'CNN':<15} {cnn_params:>12,} "
      f"{max(cnn_history['test_acc']):>14.1f}%")
print("=" * 55)

# ========== Part F ==========
def per_class_accuracy(model, loader):
    model.eval()
    correct = [0] * 10
    total = [0] * 10
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, preds = model(images).max(1)
            for c in range(10):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c] += mask.sum().item()
    return [100.0 * correct[c] / total[c] for c in range(10)]

cnn_class_acc = per_class_accuracy(cnn_model, test_loader)
dense_class_acc = per_class_accuracy(dense_model, test_loader)

x = np.arange(len(CLASSES))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width / 2, dense_class_acc, width,
       label='Dense', color='#E53935', alpha=0.8)
ax.bar(x + width / 2, cnn_class_acc, width,
       label='CNN', color='#1A237E', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=11)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100)
ax.set_title('Per-Class Accuracy: Dense vs CNN', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=150)
plt.show()