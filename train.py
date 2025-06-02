import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models.attention_unet import AttentionUNet
from utils.dataset import get_loaders
from utils.losses import CombinedLoss
from utils.metrics import (
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    precision_recall,
    AverageMeter
)
from utils.visualization import (
    visualize_prediction,
    generate_gradcam,
    plot_training_curves,
    save_predictions_as_images
)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    epoch_loss = AverageMeter()
    epoch_dice = AverageMeter()
    epoch_iou = AverageMeter()
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        
        # Forward pass
        if scaler is not None:  # Using CUDA
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # Using CPU
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
        
        # Update progress
        epoch_loss.update(loss.item(), images.size(0))
        epoch_dice.update(dice.item(), images.size(0))
        epoch_iou.update(iou.item(), images.size(0))
        
        progress_bar.set_postfix({
            'loss': f'{epoch_loss.avg:.4f}',
            'dice': f'{epoch_dice.avg:.4f}',
            'iou': f'{epoch_iou.avg:.4f}'
        })
    
    return epoch_loss.avg, epoch_dice.avg, epoch_iou.avg

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = AverageMeter()
    val_dice = AverageMeter()
    val_iou = AverageMeter()
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            
            val_loss.update(loss.item(), images.size(0))
            val_dice.update(dice.item(), images.size(0))
            val_iou.update(iou.item(), images.size(0))
    
    return val_loss.avg, val_dice.avg, val_iou.avg

def main():
    # Hyperparameters
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 2  # Reduced batch size
    NUM_EPOCHS = 30  # Changed from 100 to 30
    NUM_WORKERS = 0  # Keep 0 for CPU
    IMAGE_HEIGHT = 224  # Reduced image size
    IMAGE_WIDTH = 224   # Reduced image size
    PIN_MEMORY = DEVICE == "cuda"
    LOAD_MODEL = False
    
    # Enable garbage collection
    import gc
    gc.enable()
    
    # Directories
    TRAIN_IMG_DIR = "Kvasir-SEG/train/images"
    TRAIN_MASK_DIR = "Kvasir-SEG/train/masks"
    VAL_IMG_DIR = "Kvasir-SEG/val/images"
    VAL_MASK_DIR = "Kvasir-SEG/val/masks"
    
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    
    # Create directories for results
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved_predictions", exist_ok=True)
    
    # Initialize model, optimizer, criterion
    model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Use Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )
    
    criterion = CombinedLoss()
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    # Initialize logging
    writer = SummaryWriter('runs/polyp_segmentation')
    
    # Training loop
    best_val_dice = 0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_ious, val_ious = [], []
    
    try:
        for epoch in range(NUM_EPOCHS):
            print(f"\nStarting epoch {epoch+1}/{NUM_EPOCHS}")
            
            train_loss, train_dice, train_iou = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, epoch, scaler
            )
            
            val_loss, val_dice, val_iou = validate(
                model, val_loader, criterion, DEVICE
            )
            
            # Learning rate scheduling
            scheduler.step(val_dice)
            
            # Log metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            writer.add_scalar('Dice/val', val_dice, epoch)
            writer.add_scalar('IoU/train', train_iou, epoch)
            writer.add_scalar('IoU/val', val_iou, epoch)
            
            # Save metrics for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_dices.append(train_dice)
            val_dices.append(val_dice)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
            print(f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                print(f'New best model saved with Dice score: {best_val_dice:.4f}')
            
            # Plot and save training curves
            if epoch % 5 == 0:  # Reduced frequency of saving plots
                try:
                    metrics_dict = {
                        'Dice': val_dices,
                        'IoU': val_ious
                    }
                    plot_training_curves(
                        train_losses, val_losses, metrics_dict,
                        save_path='training_curves.png'
                    )
                except Exception as e:
                    print(f"Warning: Failed to save training curves: {str(e)}")
            
            # Save sample predictions
            if epoch % 10 == 0:
                try:
                    save_predictions_as_images(
                        val_loader, model, folder="saved_predictions",
                        device=DEVICE
                    )
                except Exception as e:
                    print(f"Warning: Failed to save predictions: {str(e)}")
            
            # Clear memory
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save the model even if training is interrupted
        torch.save(model.state_dict(), 'checkpoints/interrupted_model.pth')
    except Exception as e:
        print(f"\nTraining interrupted due to error: {str(e)}")
        # Save the model even if training is interrupted
        torch.save(model.state_dict(), 'checkpoints/interrupted_model.pth')
        raise  # Re-raise the exception for debugging
        
    writer.close()
    print('Training completed!')

if __name__ == "__main__":
    main() 