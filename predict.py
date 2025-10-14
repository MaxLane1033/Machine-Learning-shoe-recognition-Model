"""
Prediction script for shoe brand classification
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from model import create_model

def load_model(model_path, device):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    
    model = create_model(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, classes

def preprocess_image(image_path, image_size=224):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image

def predict(model, image_tensor, classes, device):
    """Make prediction on a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict shoe brand from image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model, classes = load_model(args.model_path, device)
    print(f'Model loaded. Classes: {classes}')
    
    # Preprocess image
    print('Preprocessing image...')
    image_tensor = preprocess_image(args.image_path, args.image_size)
    
    # Make prediction
    print('Making prediction...')
    predicted_class, confidence, all_probabilities = predict(model, image_tensor, classes, device)
    
    # Display results
    print(f'\nPrediction Results:')
    print(f'Predicted Brand: {predicted_class}')
    print(f'Confidence: {confidence:.2%}')
    print(f'\nAll Probabilities:')
    for i, (class_name, prob) in enumerate(zip(classes, all_probabilities)):
        print(f'{class_name}: {prob:.2%}')

if __name__ == '__main__':
    main()
