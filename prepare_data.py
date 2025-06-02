import os
import shutil
import random
from pathlib import Path

def create_train_val_split(base_dir, train_ratio=0.8):
    """
    Split the dataset into training and validation sets.
    """
    # Create directories
    for split in ['train', 'val']:
        for dtype in ['images', 'masks']:
            os.makedirs(os.path.join(base_dir, split, dtype), exist_ok=True)
    
    # Get all image files
    image_files = os.listdir(os.path.join(base_dir, 'images'))
    random.shuffle(image_files)
    
    # Calculate split
    n_train = int(len(image_files) * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    # Copy files to respective directories
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for img_file in files:
            # Copy image
            src_img = os.path.join(base_dir, 'images', img_file)
            dst_img = os.path.join(base_dir, split, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding mask
            src_mask = os.path.join(base_dir, 'masks', img_file)
            dst_mask = os.path.join(base_dir, split, 'masks', img_file)
            shutil.copy2(src_mask, dst_mask)
    
    print(f"Dataset split complete:")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

if __name__ == "__main__":
    base_dir = "Kvasir-SEG"
    create_train_val_split(base_dir) 