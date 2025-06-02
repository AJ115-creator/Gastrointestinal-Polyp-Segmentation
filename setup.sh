#!/bin/bash

echo "Setting up virtual environment for Polyp Segmentation project..."

# Check if python3-venv is installed
if ! dpkg -l | grep -q python3-venv; then
    echo "Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install requirements
echo "Activating virtual environment and installing requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed! To activate the virtual environment, run:"
echo "source venv/bin/activate" 