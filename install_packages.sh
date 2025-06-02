#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install packages one by one with latest compatible versions
pip install torch==2.7.0
pip install torchvision==0.22.0
pip install numpy==2.2.6
pip install opencv-python==4.11.0.86
pip install albumentations==2.0.8
pip install scikit-learn==1.6.1
pip install matplotlib==3.10.3
pip install tqdm==4.67.1
pip install pandas==2.2.3
pip install Pillow==11.2.1
pip install tensorboard==2.19.0
pip install grad-cam

echo "Package installation completed!" 