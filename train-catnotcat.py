import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import WeightedRandomSampler
from PIL import ImageFile
from collections import Counter
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Test Accuracy: 0.9759
# Test Precision: 0.9699
# Test Recall: 0.9945
# Test F1-Score: 0.9820

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define image size and batch size
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5

    # Data augmentation and rescaling for training data
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Only rescaling for validation and test data
    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets with ImageFolder
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transform)
    validate_dataset = datasets.ImageFolder(root='dataset/validation', transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=val_test_transform)

    # Print the number of images in each dataset
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(validate_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    # Calculate weights for each class
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.targets]

    # Create a sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Use the sampler in the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)

    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load a pre-trained model and modify it for binary classification
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Cat and Not Cat
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Early stopping and model checkpoint
    best_val_loss = float('inf')
    early_stopping_counter = 0
    model_checkpoint_path = 'best_cat_not_cat_classifier_model.pth'

    # Training and validation function
    def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(loader)}], Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}')

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping and model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_checkpoint_path)
            print(f"Saved best model at epoch {epoch+1}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_checkpoint_path))

    # Evaluate the model
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()
