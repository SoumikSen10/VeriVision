# resnext_model_script.py


# Imports
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# requires 2 agruments:- image path (frame) and pretrained resnext model path
def process_frame_with_resnext(image_path, model_path):
    print("Processing frame with ResNeXt model...")

    # Load ResNeXt model architecture 101 model
    resnext = models.resnext101_32x8d()  # Initialize the model
    resnext.load_state_dict(torch.load(model_path))  # Load custom model weights

    # Remove the fully connected layer
    modules = list(resnext.children())[:-1]
    feature_extractor = nn.Sequential(*modules)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to 224x224 for ResNeXt input
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(  # Normalize based on ResNeXt normalization stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path)  # Open image
    input_tensor = transform(image).unsqueeze(0)  # Apply transforms and add batch dimensions

    # Extract features
    with torch.no_grad():  # Disable gradient calculation
        features = feature_extractor(input_tensor)

    # Flatten the features to a 1D vector
    features = features.view(features.size(0), -1)  # Shape: (1, 2048)

    # Reshape to (sequence_length, input_size) format, i.e., (1, 2048)
    features_for_lstm = features.unsqueeze(0)  # Shape: (1, 1, 2048) -> Sequence length = 1

    print(f"Extracted feature vector shape: {features_for_lstm.shape}")
    print("Extracted feature vector:", features_for_lstm)

    return features_for_lstm


# Example usage:
if __name__ == "__main__":
    image_path = "image.jpeg"  # Replace with your image file path
    model_path = "resnext101_32x8d-8ba56ff5.pth"  # Path to the .pth model

    # Process the frame and get features
    features_for_lstm = process_frame_with_resnext(image_path, model_path)
