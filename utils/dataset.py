import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, phase='train'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.phase = phase
        self.image_files = sorted(os.listdir(images_dir))
        
        if transform is None:
            self.transform = self.get_transforms(phase)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Read image and mask
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is 3 channels
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3:
            raise ValueError(f"Image at {img_path} has {image.shape[2]} channels, expected 3")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
        
        # Normalize mask to binary
        mask = mask / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0).float(),
            'image_path': img_path
        }

    @staticmethod
    def get_transforms(phase):
        if phase == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-45, 45),
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        p=0.5
                    ),
                    A.GridDistortion(num_steps=5, p=0.5),
                    A.OpticalDistortion(
                        distort_limit=0.5,
                        p=0.5
                    ),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5
                    ),
                ], p=0.3),
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:  # val/test transforms
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = PolypDataset(
        images_dir=train_dir,
        masks_dir=train_maskdir,
        phase='train'
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PolypDataset(
        images_dir=val_dir,
        masks_dir=val_maskdir,
        phase='val'
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader 