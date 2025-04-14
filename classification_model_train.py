import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
])
dataset = datasets.ImageFolder(root="dataset", transform=transform)

TRAIN_SIZE = int(0.8 * len(dataset))
VAL_SIZE = len(dataset) - TRAIN_SIZE
BATCH_SIZE = 32
NUM_EPOCHS = 5

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

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = torch.tensor(0.0)

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / TRAIN_SIZE
    epoch_acc = running_corrects / TRAIN_SIZE
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")

    # Validation phase.
    model.eval()
    val_loss = 0.0
    val_corrects = 0
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
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

print("Finetuning complete!")

# Test on one validation image and display it
model.eval()
test_inputs, test_labels = next(iter(val_loader))
test_input = test_inputs[0].unsqueeze(0).to(device)  # Get first image and add batch dimension
test_label = test_labels[0]

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
    output = model(test_input)
    pred = output.argmax(dim=1).item()

print(f"\nTest image actual label: {dataset.classes[test_label]}")
print(f"Model prediction: {dataset.classes[pred]}")
