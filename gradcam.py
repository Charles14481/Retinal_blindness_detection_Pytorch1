# Importing all packages 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image, ImageFile
import json
from torch.optim import lr_scheduler
import random
import os
import sys
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

print('Imported packages')
def init_model():
    device = torch.device("cpu")
    model = models.resnet152(pretrained=False)
    num_ftrs = model.fc.in_features
    out_ftrs = 5
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),nn.ReLU(),nn.Linear(512,out_ftrs),nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.00001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.to(device);
    # to unfreeze more layers


    for name,child in model.named_children():
        if name in ['layer2','layer3','layer4','fc']:
            #print(name + 'is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            #print(name + 'is frozen')
            for param in child.parameters():
                param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.000001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return model, optimizer

def load_model(path):
    model, optimizer = init_model()
    checkpoint = torch.load(path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def inference(model, file, transform, classes):
    file = Image.open(file).convert('RGB')
    img = transform(file).unsqueeze(0)
    print('Transforming your image...')
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        print('Passing your image to the model....')
        out = model(img.to(device))
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        probability = top_p.item()
        print("Predicted Severity Value: ", value)
        print("class is: ", classes[value])
        print("probability is: ", probability)
        print('Your image is printed:')
        return {
            "class": classes[value],
            "value": value,
            "probability": probability
        }
        # plt.imshow(np.array(file))
        # plt.show()

def get_gradcam(model, img_tensor, target_layer, device):
    """
    Generate GradCAM visualization for the input image
    """
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    targets = [ClassifierOutputTarget(0)]  # We'll use the predicted class
    grayscale_cam = cam(input_tensor=img_tensor.to(device), targets=targets)
    return grayscale_cam[0, :]

def visualize_gradcam(image_path, model, transform, device):
    """
    Visualize the original image with GradCAM overlay
    """
    # Load and transform the image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(device))
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        predicted_class = top_class.item()
    
    # Get GradCAM
    target_layer = model.layer4[-1]  # Using the last layer of ResNet
    grayscale_cam = get_gradcam(model, input_tensor, target_layer, device)
    
    # Convert image to numpy array for visualization
    img_np = np.array(img)
    img_np = cv2.resize(img_np, (224, 224))
    img_np = img_np / 255.0
    
    # Create visualization
    visualization = show_cam_on_image(img_np, grayscale_cam)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # GradCAM visualization
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'GradCAM Visualization\nPredicted Class: {predicted_class}')
    plt.axis('off')
    
    # plt.show()
    plt.savefig('heatmap.png')

def test(path):
    model = load_model('/home/charles/project_dr/Retinal_blindness_detection_Pytorch/classifier.pt')
    print("Model loaded Successfully")
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    device = torch.device("cpu")
    dict = inference(model, path, test_transforms, classes)
    visualize_gradcam(path, model, test_transforms, device)
    return dict

if __name__ == '__main__':
    test_dir = '/home/charles/project_dr/Retinal_blindness_detection_Pytorch/sampleimages'
    folders = os.listdir(test_dir)
    for num in range(len(folders)):
        path = test_dir+"/"+folders[num]
        print(path)
        test(path)