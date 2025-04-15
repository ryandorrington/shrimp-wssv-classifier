import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os
import zipfile

torch.manual_seed(21)

# Check if dataset directory exists
if not os.path.exists("dataset"):
    print("Dataset directory not found. Checking for dataset.zip...")
    if os.path.exists("dataset.zip"):
        print("Found dataset.zip. Extracting...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")
    else:
        print("Error: Neither 'dataset' directory nor 'dataset.zip' found.")
        print("Please ensure you have either the dataset folder or the dataset.zip file.")
        exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
])
dataset = datasets.ImageFolder(root="dataset", transform=transform)

TRAIN_SIZE: int = int(0.8 * len(dataset))
VAL_SIZE: int = len(dataset) - TRAIN_SIZE
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 1

train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained ResNet-18.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace the final layer to output 2 classes.
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

print("Using device:", device)
print("Classes:", dataset.classes)
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Create lists to store loss and accuracy history
train_loss_history: list = []
val_loss_history: list = []
train_acc_history: list = []
val_acc_history: list = []

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss: float = 0.0
    running_corrects: torch.Tensor = torch.tensor(0.0)

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds: torch.Tensor = outputs.argmax(dim=1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss: float = running_loss / TRAIN_SIZE
    epoch_acc: torch.Tensor = running_corrects / TRAIN_SIZE
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())

    # Validation phase.
    model.eval()
    val_loss: float = 0.0
    val_corrects: int = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / VAL_SIZE
    val_acc = val_corrects / VAL_SIZE
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc.item())

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Time: {epoch_time:.2f}s")

# Calculate total training time
training_time = time.time() - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Finetuning complete! Total training time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS+1), train_loss_history, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS+1), val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS+1), [acc*100 for acc in train_acc_history], label="Train Accuracy")
plt.plot(range(1, NUM_EPOCHS+1), [acc*100 for acc in val_acc_history], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Training and Validation Accuracy")

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# Test on one validation image and display it
model.eval()
test_inputs: torch.Tensor
test_labels: torch.Tensor
test_inputs, test_labels = next(iter(val_loader))
test_input: torch.Tensor = test_inputs[0].unsqueeze(0).to(device)  # Get first image and add batch dimension
test_label: int = test_labels[0].item()

# Convert tensor back to image for display
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
img = inv_normalize(test_inputs[0]).permute(1, 2, 0).cpu().numpy()
plt.imshow(img)
plt.axis('off')
plt.show()

with torch.no_grad():
    output: torch.Tensor = model(test_input)
    pred: float = output.argmax(dim=1).item()

print(f"\nTest image actual label: {dataset.classes[test_label]}")
print(f"Model prediction: {dataset.classes[pred]}")
