# Fashion MNIST Image Classification Web App

This project is a Streamlit-based web application that allows users to upload images and classify them into one of the categories from the Fashion MNIST dataset. It uses a Convolutional Neural Network (CNN) model trained on the Fashion MNIST dataset to predict the class of clothing images such as T-shirts, trousers, dresses, and more.

### Features:
- **Image Upload**: Users can upload their own clothing images to be classified.
- **Sample Images**: A set of sample images from the Fashion MNIST dataset is available for users to try. These images can be downloaded for offline testing.
- **Real-time Predictions**: The model predicts the class of the uploaded image and displays the result in real-time.
- **Download Option**: Sample images can be downloaded directly from the website.

### Technologies Used:
- **Streamlit**: For building the interactive web application interface.
- **PyTorch**: For the machine learning model (Convolutional Neural Network).
- **Fashion MNIST Dataset**: A dataset of 60,000 28x28 grayscale images of 10 fashion categories.
- **PIL & torchvision**: For image manipulation and preprocessing.

---

## How to Run Locally

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/fashion-mnist-web-app.git
   cd fashion-mnist-web-app
