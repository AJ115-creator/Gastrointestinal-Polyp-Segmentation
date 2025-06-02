import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def visualize_prediction(image, mask, prediction, save_path=None):
    """
    Visualize the original image, ground truth mask, and prediction mask side by side
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # If channels first, transpose to channels last
            image = image.transpose(1, 2, 0)
        elif len(image.shape) == 2:  # If grayscale, expand to RGB
            image = np.stack([image] * 3, axis=-1)
        # Denormalize
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    if isinstance(prediction, torch.Tensor):
        prediction = torch.sigmoid(prediction).cpu().numpy()
    if len(prediction.shape) > 2:
        prediction = prediction.squeeze()
    prediction = (prediction > 0.5).astype(np.float32)
    plt.imshow(prediction, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_gradcam(model, image, target_layer):
    """
    Generate Grad-CAM visualization for the model's attention
    """
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    
    # Generate grayscale CAM
    grayscale_cam = cam(input_tensor=image.unsqueeze(0))
    grayscale_cam = grayscale_cam[0]
    
    # Convert input image for visualization
    image_for_cam = image.cpu().numpy().transpose(1, 2, 0)
    image_for_cam = image_for_cam * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_for_cam = np.clip(image_for_cam, 0, 1)
    
    # Overlay CAM on image
    cam_image = show_cam_on_image(image_for_cam, grayscale_cam, use_rgb=True)
    
    return cam_image

def plot_training_curves(train_losses, val_losses, metrics_dict, save_path=None):
    """
    Plot training and validation curves
    """
    n_metrics = len(metrics_dict) + 1
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot metrics
    for idx, (metric_name, metric_values) in enumerate(metrics_dict.items(), 1):
        axes[idx].plot(metric_values)
        axes[idx].set_title(f'{metric_name} Curve')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_predictions_as_images(loader, model, folder="saved_predictions", device="cuda"):
    """
    Save model predictions as images
    """
    model.eval()
    for idx, batch in enumerate(loader):
        x = batch["image"].to(device)  # Shape: [B, C, H, W]
        y = batch["mask"]  # Shape: [B, 1, H, W]
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))  # Shape: [B, 1, H, W]
        
        # Process each image in the batch
        for i in range(x.size(0)):
            # Get single image, mask and prediction
            single_image = x[i]  # Shape: [C, H, W]
            single_mask = y[i].cpu()  # Shape: [1, H, W]
            single_pred = preds[i].cpu()  # Shape: [1, H, W]
            
            # Save the predictions
            save_path = f"{folder}/pred_{idx}_{i}.png"
            visualize_prediction(
                image=single_image.cpu(),
                mask=single_mask,
                prediction=single_pred,
                save_path=save_path
            )
        
        # Only save first few batches to avoid too many images
        if idx >= 4:  # Save predictions for first 5 batches only
            break 