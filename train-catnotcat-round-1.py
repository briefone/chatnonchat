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

def get_model(model_name, weights):
    if model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights=weights)
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python train-catnotcat-round-1.py <model_name> <optimizer_name[Adam, RMSprop, SGD]> <scheduler_name[StepLR, CosineAnnealingLR, ReduceLROnPlateau]>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    optimizer_name = sys.argv[2] if len(sys.argv) > 2 else "Adam"
    scheduler_name = sys.argv[3] if len(sys.argv) > 3 else "StepLR"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set image size based on model
    if model_name == 'inception_v3':
        IMAGE_SIZE = (299, 299)
    else:
        IMAGE_SIZE = (224, 224)
        
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
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

    # Handle different model weight classes
    weight_classes = {
        'wide_resnet50_2': models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
        'densenet121': models.DenseNet121_Weights.IMAGENET1K_V1,
        'mobilenet_v2': models.MobileNet_V2_Weights.IMAGENET1K_V1,
        'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'vit_b_16': models.ViT_B_16_Weights.IMAGENET1K_V1,
        'alexnet': models.AlexNet_Weights.IMAGENET1K_V1,
        'vgg16': models.VGG16_Weights.IMAGENET1K_V1,
        'inception_v3': models.Inception_V3_Weights.IMAGENET1K_V1,
        'resnext50_32x4d': models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    }

    weights = weight_classes.get(model_name, None)
    if weights is None:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = get_model(model_name, weights)
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
    
    scheduler_params = all_scheduler_params[scheduler_name]
    
    optimizer = get_optimizer(model, optimizer_name, LEARNING_RATE)
    scheduler = get_scheduler(optimizer, scheduler_name, scheduler_params)

    scaler = GradScaler()

    best_val_loss = float('inf')
    early_stopping_counter = 0
    model_checkpoint_path = f'catnotcat_{model_name}_{optimizer_name}_{scheduler_name}_round_1.pth'

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

    # Validate on test set
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()
