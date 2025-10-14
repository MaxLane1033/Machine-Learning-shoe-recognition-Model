# Machine Learning Shoe Recognition Model

A PyTorch-based image classification model that can distinguish between Nike and Adidas shoes. This project demonstrates computer vision techniques for brand recognition in footwear.

## Features

- Convolutional Neural Network (CNN) architecture
- Data augmentation for improved generalization
- Training and validation loops with metrics tracking
- Model saving and loading functionality
- Simple inference script for testing

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib
- numpy

## Installation

```bash
pip install torch torchvision pillow matplotlib numpy
```

## Usage

### Training the Model

1. Prepare your dataset with the following structure:
```
data/
├── train/
│   ├── nike/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── adidas/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── nike/
    └── adidas/
```

2. Run the training script:
```bash
python train.py --data_dir data --epochs 50 --batch_size 32
```

### Making Predictions

```bash
python predict.py --model_path model.pth --image_path test_image.jpg
```

## Model Architecture

The model uses a custom CNN architecture with:
- 3 convolutional layers with ReLU activation
- Max pooling layers for downsampling
- Dropout for regularization
- Fully connected layers for classification

## Training Parameters

- **Learning Rate**: 0.001 (adjustable)
- **Batch Size**: 32 (adjustable)
- **Epochs**: 50 (adjustable)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Data Augmentation

The model includes data augmentation techniques:
- Random horizontal flip
- Random rotation
- Color jitter
- Random crop and resize

## Performance

The model achieves good accuracy on the validation set. Performance can be improved by:
- Using more training data
- Implementing transfer learning with pre-trained models
- Fine-tuning hyperparameters
- Using more sophisticated data augmentation

## File Structure

- `model.py`: CNN model definition
- `train.py`: Training script
- `predict.py`: Inference script
- `utils.py`: Utility functions
- `requirements.txt`: Dependencies

## License

This project is open source and available under the MIT License.
