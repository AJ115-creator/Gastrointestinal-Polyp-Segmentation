import os
import torch
import cv2
import numpy as np
from models.attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.visualization import visualize_prediction, generate_gradcam
import matplotlib.pyplot as plt

class PolypSegmentationInference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = AttentionUNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image):
        # Convert BGR to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Apply transformations
        transformed = self.transform(image=image)
        return transformed['image']
    
    def predict(self, image, return_prob=False):
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float()
        
        # Convert to numpy
        if return_prob:
            return pred[0].cpu().numpy(), prob[0].cpu().numpy()
        return pred[0].cpu().numpy()
    
    def predict_with_visualization(self, image, save_path=None):
        # Get prediction
        pred_mask, pred_prob = self.predict(image, return_prob=True)
        
        # Generate Grad-CAM
        input_tensor = self.preprocess_image(image).to(self.device)
        cam_image = generate_gradcam(
            self.model,
            input_tensor,
            target_layer=self.model.encoder[-1]  # Last encoder layer
        )
        
        # Prepare visualization
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Prediction probability
        plt.subplot(142)
        plt.imshow(pred_prob.squeeze(), cmap='jet')
        plt.title('Prediction Probability')
        plt.colorbar()
        plt.axis('off')
        
        # Binary prediction
        plt.subplot(143)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title('Binary Prediction')
        plt.axis('off')
        
        # Grad-CAM visualization
        plt.subplot(144)
        plt.imshow(cam_image)
        plt.title('Grad-CAM Visualization')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return pred_mask, cam_image

def main():
    # Example usage
    model_path = 'checkpoints/best_model.pth'
    image_path = 'path/to/test/image.jpg'
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference
    inferencer = PolypSegmentationInference(model_path)
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get prediction with visualization
    save_path = os.path.join(output_dir, 'prediction_visualization.png')
    pred_mask, cam_image = inferencer.predict_with_visualization(
        image, save_path=save_path
    )
    
    # Save binary prediction
    pred_path = os.path.join(output_dir, 'prediction_mask.png')
    cv2.imwrite(pred_path, (pred_mask.squeeze() * 255).astype(np.uint8))
    
    # Save Grad-CAM visualization
    cam_path = os.path.join(output_dir, 'gradcam.png')
    cv2.imwrite(cam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 