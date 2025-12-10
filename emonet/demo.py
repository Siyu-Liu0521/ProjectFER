from pathlib import Path
import argparse

import torch
from torch import nn
from skimage import io

from emonet.models import EmoNet

import cv2

torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
parser.add_argument('--image_path', type=str, default="images/example.png", help='Path to a face image.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
image_size = 256
emotion_classes = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}
image_path = Path(__file__).parent / args.image_path

# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

print(f'Testing on a single image')
print(f'------------------------')
# Load image in RGB format
image_rgb = io.imread(image_path)[:,:,:3]

# Resize image to (256,256)
image_rgb = cv2.resize(image_rgb, (image_size, image_size))

# Load image into a tensor: convert to RGB, and put the tensor in the [0;1] range
image_tensor = torch.Tensor(image_rgb).permute(2,0,1).to(device)/255.0

with torch.no_grad():
    output = net(image_tensor.unsqueeze(0))
    
    # Calculate confidence: probability of the predicted emotion class
    emotion_probs = nn.functional.softmax(output["expression"], dim=1)
    predicted_emotion_class = torch.argmax(emotion_probs, dim=1).cpu().item()
    confidence = emotion_probs[0, predicted_emotion_class].cpu().item()
    
    # Calculate quality: geometric-style (multiplicative) quality
    # Quality is a geometric mean of:
    # 1. Confidence in emotion prediction (0-1)
    # 2. Heatmap quality (normalized intensity of landmarks)
    heatmap = output["heatmap"]
    heatmap_max = torch.max(heatmap).cpu().item()
    
    # Heatmap is sparse (mostly zeros, only landmark locations have values)
    # So we use max value and non-zero statistics for quality assessment
    heatmap_nonzero = heatmap[heatmap > 0.01]  # Get non-zero values (threshold to ignore noise)
    if len(heatmap_nonzero) > 0:
        heatmap_nonzero_mean = torch.mean(heatmap_nonzero).cpu().item()
        # Use combination of max and non-zero mean for robust quality
        # Heatmap values typically range [0, ~1-2] after normalization
        heatmap_quality_max = min(heatmap_max / 1.5, 1.0)  # Normalize by typical max
        heatmap_quality_mean = min(heatmap_nonzero_mean / 0.5, 1.0)  # Normalize by typical non-zero mean
        heatmap_quality = 0.7 * heatmap_quality_max + 0.3 * heatmap_quality_mean
    else:
        # Fallback if no significant heatmap values
        heatmap_quality = min(heatmap_max / 1.5, 1.0)
    
    # Geometric-style quality: multiplicative (both factors must be high)
    # Using geometric mean: sqrt(confidence * heatmap_quality)
    # This ensures quality is high only when both confidence and heatmap are high
    quality = (confidence * heatmap_quality) ** 0.5
    
    # Expected output on example image: Predicted Emotion Happy - valence 0.064 - arousal 0.143
    print(f"Predicted Emotion: {emotion_classes[predicted_emotion_class]}")
    print(f"Valence: {output['valence'].clamp(-1.0,1.0).cpu().item():.3f}")
    print(f"Arousal: {output['arousal'].clamp(-1.0,1.0).cpu().item():.3f}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Heatmap Quality: {heatmap_quality:.3f}")
    print(f"Quality: {quality:.3f}")