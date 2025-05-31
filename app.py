import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Define the class names
class_names = ['Brain Glioma', 'Brain Meningioma', 'Brain Tumor']

# Load the trained model
def load_model(model_path):
    model = models.densenet121(weights=None)
    
    # Modify classifier 
    in_features = model.classifier.in_features
    modified_classification_layer = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    )
    model.classifier = modified_classification_layer
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Custom transform to ensure image is RGB
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# Normalization values used during training
mean = [0.1543, 0.1543, 0.1543]
std = [0.1668, 0.1668, 0.1668]

# Image preprocessing
def get_transforms():
    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# Load the model
model_path = 'model/best_model.pth' 
model = load_model(model_path)
transform = get_transforms()

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # DEBUG OUTPUT
        print("Raw model outputs:", outputs[0])
        print("Probabilities:", probabilities)
        predicted_idx = torch.argmax(probabilities).item()
        print(f"Predicted class index: {predicted_idx}")
        print(f"Predicted class name: {class_names[predicted_idx]}")
        
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return confidences
examples= ['examples/brain_tumor.jpg',
           'examples/brain_glioma.jpg',
           'examples/brain_menin.jpg'
           ]
# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Brain Tumor MRI Classifier",
    description="Upload an MRI image to detect whether it is a Brain Tumor, Glioma, or Meningioma.",
    theme=gr.themes.Soft(),
    examples=examples
)

# Launch app
interface.launch(debug=True,share=True)