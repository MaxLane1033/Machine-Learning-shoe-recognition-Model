"""
Utility functions for shoe classification model
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def visualize_predictions(model, data_loader, classes, device, num_images=8):
    """Visualize model predictions on a batch of images"""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Move to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    probabilities = probabilities.cpu()
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        # Denormalize image
        img = images[i]
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        img_np = img.numpy().transpose(1, 2, 0)
        
        # Plot image
        axes[i].imshow(img_np)
        axes[i].axis('off')
        
        # Set title with prediction info
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        confidence = probabilities[i][predicted[i]] * 100
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, data_loader, classes, device):
    """Plot confusion matrix for model evaluation"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def calculate_accuracy(model, data_loader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def save_model_info(model, save_path, classes, training_info=None):
    """Save model information to a text file"""
    with open(save_path.replace('.pth', '_info.txt'), 'w') as f:
        f.write("Model Information\n")
        f.write("================\n\n")
        
        f.write(f"Classes: {classes}\n")
        f.write(f"Number of classes: {len(classes)}\n")
        f.write(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n\n")
        
        if training_info:
            f.write("Training Information\n")
            f.write("==================\n")
            for key, value in training_info.items():
                f.write(f"{key}: {value}\n")

def load_model_info(model_path):
    """Load model information from text file"""
    info_path = model_path.replace('.pth', '_info.txt')
    try:
        with open(info_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Model info file not found"

if __name__ == "__main__":
    print("Utility functions for shoe classification model")
    print("Available functions:")
    print("- visualize_predictions: Show model predictions on images")
    print("- plot_confusion_matrix: Create confusion matrix")
    print("- calculate_accuracy: Calculate model accuracy")
    print("- save_model_info: Save model information")
    print("- load_model_info: Load model information")
