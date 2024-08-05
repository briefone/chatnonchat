import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch.utils.data import WeightedRandomSampler
from PIL import ImageFile, ImageSequence
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tkinter as tk
from tkinter import filedialog

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AlbumentationsTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        if image.format == 'GIF':
            image = ImageSequence.Iterator(image)[0]
        elif image.format == 'MPO':
            image = ImageSequence.Iterator(image)[0]
        
        if image.mode == 'P' and 'transparency' in image.info:
            image = image.convert('RGBA')
        elif image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGB')
            
        image = np.array(image)
        augmented = self.transforms(image=image)
        return augmented['image']
  
def get_model(model_name):
    if model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet18':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet34':
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet50':
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'densenet121':
        model = models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16()
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)
    elif model_name == 'alexnet':
        model = models.alexnet()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
    elif model_name == 'vgg16':
        model = models.vgg16()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
    elif model_name == 'inception_v3':
        model = models.inception_v3()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model
  
def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer

def get_scheduler(optimizer, scheduler_name, scheduler_params):
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    return scheduler

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler, NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Inception_v3 outputs a tuple, we only need the first output
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

def parse_model_name(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    known_models = [
        'wide_resnet50_2', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'mobilenet_v2', 
        'efficientnet_b0', 'vit_b_16', 'alexnet', 'vgg16', 'inception_v3', 
        'resnext50_32x4d'
    ]
    for model_name in known_models:
        if model_name in filename:
            return model_name
    raise ValueError(f"Unsupported model name in checkpoint path: {checkpoint_path}")

def main():
    # Open file picker dialog to select the model checkpoint
    root = tk.Tk()
    root.withdraw()
    first_round_checkpoint_path = filedialog.askopenfilename(title="Select the model checkpoint from the first round")

    if not first_round_checkpoint_path:
        print("No file selected, exiting.")
        sys.exit(1)
    
    # Extract model name from the file name
    # model_name = os.path.basename(first_round_checkpoint_path).split('_')[1]
    
    model_name = parse_model_name(first_round_checkpoint_path)
    print(f"Model architecture: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set image size based on model
    if model_name == 'inception_v3':
        IMAGE_SIZE = (299, 299)
    else:
        IMAGE_SIZE = (224, 224)
        
    BATCH_SIZE = 32
    NUM_EPOCHS = 30  # Number of additional epochs
    LEARNING_RATE = 0.0001  # Reduced learning rate for fine-tuning
    EARLY_STOPPING_PATIENCE = 5

    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        A.RandomResizedCrop(IMAGE_SIZE[0], IMAGE_SIZE[1], scale=(0.8, 1.0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_test_transform = A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.CenterCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = datasets.ImageFolder(root='dataset/train', transform=AlbumentationsTransform(train_transform))
    validate_dataset = datasets.ImageFolder(root='dataset/validation', transform=AlbumentationsTransform(val_test_transform))
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=AlbumentationsTransform(val_test_transform))

    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(validate_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")

    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_model(model_name)
    model.load_state_dict(torch.load(first_round_checkpoint_path))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # optimizers_to_test = ["Adam", "RMSprop", "SGD"]
    # schedulers_to_test = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    
    all_scheduler_params = {
        "StepLR": {"step_size": 10, "gamma": 0.1},
        "CosineAnnealingLR": {"T_max": 10},
        "ReduceLROnPlateau": {"mode": 'min', "factor": 0.1, "patience": 3, "verbose": True}
    }
    
    optimizer_name = sys.argv[1] if len(sys.argv) > 1 else "Adam"
    scheduler_name = sys.argv[2] if len(sys.argv) > 2 else "StepLR"
    # optimizer_name = "Adam"
    # scheduler_name = "StepLR"
    scheduler_params = all_scheduler_params[scheduler_name]
    
    optimizer = get_optimizer(model, optimizer_name, LEARNING_RATE)
    scheduler = get_scheduler(optimizer, scheduler_name, scheduler_params)

    scaler = GradScaler()

    best_val_loss = float('inf')
    early_stopping_counter = 0
    model_checkpoint_path = f'catnotcat_{model_name}_{optimizer_name}_{scheduler_name}_round_2.pth'

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, NUM_EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

    model.load_state_dict(torch.load(model_checkpoint_path))

    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()
