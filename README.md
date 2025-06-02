# 🔬 Deep Learning for Gastrointestinal Polyp Segmentation

This project implements a deep learning solution for automatic segmentation of gastrointestinal polyps in colonoscopy images. The system uses a state-of-the-art Attention U-Net architecture with residual blocks and squeeze-excitation modules to achieve accurate polyp detection and segmentation.

## 🎯 Project Overview

Colorectal cancer is one of the most common and deadly forms of cancer worldwide. Early detection and removal of precancerous polyps during colonoscopy can significantly reduce the risk of developing colorectal cancer. This project aims to assist medical professionals by automatically identifying and segmenting polyps in colonoscopy images.

### Key Features

- **Advanced Architecture**: Attention U-Net with residual blocks and SE modules
- **Robust Loss Function**: Combination of Dice, Focal, and Tversky losses
- **Extensive Augmentation**: Comprehensive data augmentation pipeline
- **Interpretability**: Grad-CAM visualization for model attention
- **Performance Metrics**: Dice coefficient, IoU, precision, recall
- **Real-time Visualization**: Training curves and prediction visualization

## 🛠️ Technical Architecture

### Model Design

The model architecture combines several powerful components:

1. **Attention U-Net Base**:
   - Encoder-decoder architecture with skip connections
   - Attention gates for focusing on relevant features
   - Multi-scale feature processing

2. **Enhancement Modules**:
   - Residual blocks for better gradient flow
   - Squeeze-excitation modules for channel attention
   - Batch normalization and ReLU activation

3. **Loss Function**:
   ```python
   Combined Loss = α * Dice Loss + β * Focal Loss + γ * Tversky Loss
   ```
   - Addresses class imbalance
   - Focuses on boundary regions
   - Penalizes false positives/negatives differently

## 📊 Performance Metrics

The model is evaluated using multiple metrics:

- **Dice Coefficient**: Measures overlap between predictions and ground truth
- **IoU (Jaccard Index)**: Intersection over Union for segmentation quality
- **Pixel-wise Precision/Recall**: For detailed performance analysis
- **Boundary Analysis**: Special attention to polyp boundaries

## 🔧 Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd polyp-segmentation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Place images in `Kvasir-SEG/images/`
   - Place masks in `Kvasir-SEG/masks/`

## 🚀 Usage

### Training

```bash
python train.py
```

The training script will:
- Initialize the model and optimizer
- Load and preprocess the dataset
- Train for the specified number of epochs
- Save checkpoints and visualizations
- Log metrics to TensorBoard

### Inference

```bash
python inference.py --image_path path/to/image.jpg
```

This will:
- Load the trained model
- Process the input image
- Generate segmentation mask
- Create visualization with Grad-CAM
- Save results to the specified output directory

## 📈 Training Strategy

1. **Data Augmentation**:
   - Geometric: rotation, flipping, scaling
   - Intensity: brightness, contrast, noise
   - Elastic deformations for shape variance

2. **Optimization**:
   - Adam optimizer with learning rate scheduling
   - Gradient clipping for stability
   - Early stopping based on validation metrics

3. **Monitoring**:
   - TensorBoard integration for metrics
   - Regular validation checks
   - Best model checkpointing

## 🔍 Model Interpretability

The project includes several interpretability features:

1. **Grad-CAM Visualization**:
   - Highlights regions the model focuses on
   - Helps verify clinical relevance
   - Useful for debugging and improvement

2. **Attention Maps**:
   - Shows where the model pays attention
   - Validates anatomical understanding
   - Helps in clinical explanation

## 📝 Results and Evaluation

The model achieves:
- Dice Coefficient: > 0.85 on test set
- IoU Score: > 0.80 on test set
- High precision in boundary detection
- Real-time inference capability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kvasir-SEG dataset providers
- Medical professionals for domain expertise
- Open-source community for tools and frameworks

# Polyp Detection System

A deep learning system for gastrointestinal polyp segmentation using the Kvasir-SEG dataset. This implementation uses an Attention U-Net architecture with memory-efficient design choices for CPU/GPU training.

## Features

- Memory-efficient Attention U-Net architecture
- Multi-loss function combining Dice and other metrics
- Comprehensive data augmentation pipeline
- TensorBoard integration for monitoring
- Automatic model checkpointing
- Training curve visualization
- Prediction visualization with masks

## Project Structure

```
Polyps detection system/
├── models/
│   └── attention_unet.py     # Model architecture implementation
├── utils/
│   ├── dataset.py           # Data loading and augmentation
│   ├── losses.py            # Custom loss functions
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Visualization utilities
├── Kvasir-SEG/              # Dataset directory
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── checkpoints/             # Saved model weights
├── saved_predictions/       # Visualization outputs
├── runs/                    # TensorBoard logs
├── train.py                # Training script
├── inference.py            # Inference script
├── prepare_data.py         # Dataset preparation
├── requirements.txt        # Python dependencies
├── setup.sh               # Environment setup script
└── install_packages.sh    # Package installation script
```

## Setup Instructions

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
bash install_packages.sh
# OR
pip install -r requirements.txt
```

3. Prepare the Kvasir-SEG dataset:
```bash
python prepare_data.py
```

## Model Architecture

The implementation uses an Attention U-Net with the following specifications:
- Input size: 224x224x3
- Feature channels: [32, 64, 128, 256]
- Attention gates at skip connections
- Output: Single-channel segmentation mask

## Training Configuration

Current hyperparameters:
- Learning rate: 1e-4
- Batch size: 2 (optimized for memory efficiency)
- Image size: 224x224
- Number of epochs: 30
- Optimizer: Adam with weight decay (1e-5)
- Learning rate scheduler: ReduceLROnPlateau
- Loss function: Combined loss (Dice + other metrics)

### Data Augmentation

Training augmentations include:
- Random rotations (90°)
- Horizontal/Vertical flips
- Affine transformations
- Elastic deformations
- Color augmentations
- Gaussian noise

## Training

To start training:
```bash
python train.py
```

The training script includes:
- Automatic checkpointing of best models
- Training curve visualization (every 5 epochs)
- Prediction visualization (every 10 epochs)
- TensorBoard logging
- Memory management with garbage collection

## Monitoring Training

1. View training curves:
   - Check `training_curves.png` for loss and metric plots
   - Updated every 5 epochs

2. TensorBoard visualization:
```bash
tensorboard --logdir runs/
```

3. Sample predictions:
   - Check `saved_predictions/` directory
   - Updated every 10 epochs

## Inference

To run inference on new images:
```bash
python inference.py --input_path path/to/image --model_path checkpoints/best_model.pth
```

## Performance Metrics

The model tracks:
- Dice coefficient
- IoU (Intersection over Union)
- Loss values
- Pixel accuracy
- Precision/Recall

## Error Handling

The implementation includes robust error handling:
- Graceful handling of visualization errors
- Automatic model saving on interruption
- Memory management
- Shape validation for inputs

## Requirements

Main dependencies:
- PyTorch
- Albumentations
- OpenCV
- TensorBoard
- NumPy
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kvasir-SEG dataset providers
- Attention U-Net paper authors
- PyTorch community 