#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# Define device
device = torch.device("cpu")

# Load the model based on model name
def get_model(model_name):
    if model_name == 'resnet18':
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
    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

# Load the model
def load_model(checkpoint_path):
    model_name = parse_model_name(checkpoint_path)
    model = get_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Parse model name from checkpoint file path
def parse_model_name(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    known_models = [
        'resnet18', 'resnet34', 'resnet50', 'densenet121', 'mobilenet_v2', 
        'efficientnet_b0', 'vit_b_16', 'alexnet', 'vgg16', 'inception_v3', 
        'resnext50_32x4d', 'wide_resnet50_2'
    ]
    for model_name in known_models:
        if model_name in filename:
            return model_name
    raise ValueError(f"Unsupported model name in checkpoint path: {checkpoint_path}")

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Predict if the image is a cat or not and return confidence level and metadata
def predict(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_class = 'Cat' if predicted.item() == 0 else 'Not Cat'
        metadata = {
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy().tolist(),
            'raw_scores': outputs.cpu().numpy().tolist()
        }
        return predicted_class, metadata

def classify_image(image_path, model_path):
    try:
        model = load_model(model_path)
        predicted_class, metadata = predict(model, image_path)
        response = {
            'class': predicted_class,
            'confidence': metadata['confidence'],
            'probabilities': metadata['probabilities'],
            'raw_scores': metadata['raw_scores']
        }
        return json.dumps(response)
    except Exception as e:
        response = {
            'error': str(e)
        }
        return json.dumps(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat or Not Cat Classifier')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    result = classify_image(args.image, args.model)
    print(result)
