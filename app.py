import streamlit as st
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
from torch import nn
import io

# List of FashionMNIST classes
class_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_list_str=  ','.join(class_list).replace(',',', ')

# Define the model class
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Load model (use the correct path to your saved model)
MODEL_PATH = Path("./Models/O1_fashionMNIST_model.pth")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
else:
    print("Model found")

model = FashionMNISTModelV1(input_shape=1, hidden_units=64, output_shape=10)

# Load weights safely
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    print("Model weights loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model weights: {e}")
    exit()
model.eval()  # Set model to evaluation mode

# Define the transformation for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor values
])

# Load the FashionMNIST dataset
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)

# Function to make predictions
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(torch.device("cpu"))  # Move to CPU (or CUDA if available)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)  # Get predicted class index
    return predicted_class.item()

# Streamlit Interface
st.write("#### Fashion MNIST Image Classification")
st.write("Upload an image of clothing to classify it into one of the categories!")
st.write(f"Classes available: {class_list_str}")

# Select a class using dropdown or tabs
selected_class = st.selectbox("Choose a Class to Display Images", class_list)

# Show images from the selected class
st.write(f"### Upload your own or use sample images of {selected_class}")
# Filter out the images of the selected class
class_idx = class_list.index(selected_class)
class_images = [img for img, label in zip(train_data.data, train_data.targets) if label == class_idx]

# Create a container for the images
cols = st.columns(5)  # Create 5 columns for images in a row

# Show 5 images from the selected class
for i in range(5):
    if i < len(class_images):
        img = class_images[i]
        img = Image.fromarray(img.numpy())
        with cols[i % 5]:  # Use the columns in a loop
            st.image(img, caption=f"{selected_class} {i+1}", width=100)
            # Create downloadable button for each sample image
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label=f"Download",
                data=buffered.getvalue(),
                file_name=f"{selected_class}_{i+1}.png",
                mime="image/png"
            )

# Function to upload and predict user-uploaded image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Display image with a smaller size
    st.image(image, caption="Uploaded Image", width=200)  # Make image smaller with width=200
    
    # Predict the class
    predicted_class = predict_image(image)
    st.markdown(f"### Predicted Class: **{class_list[predicted_class]}**", unsafe_allow_html=True)

