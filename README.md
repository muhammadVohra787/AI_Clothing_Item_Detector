# Fashion MNIST Image Classification Web App

This project is a Streamlit-based web application that allows users to upload images and classify them into one of the categories from the Fashion MNIST dataset. It uses a Convolutional Neural Network (CNN) model trained on the Fashion MNIST dataset to predict the class of clothing images such as T-shirts, trousers, dresses, and more.

## Features:
- **Image Upload**: Users can upload their own clothing images to be classified.
- **Sample Images**: A set of sample images from the Fashion MNIST dataset is available for users to try. These images can be downloaded for offline testing.
- **Real-time Predictions**: The model predicts the class of the uploaded image and displays the result in real-time.
- **Download Option**: Sample images can be downloaded directly from the website.

## Technologies Used:
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
    ```

2. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the trained model file (`fashionMNIST_model.pth`) in the `./Models/` directory. If not, you can train the model first or download the pre-trained version.

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

5. Open the app in your web browser by visiting `http://localhost:8501`.

---

## How It Works

### Upload Image:
- The web app provides an interface where users can upload an image in JPG, PNG, or JPEG format.
- The uploaded image is preprocessed and passed to the trained CNN model.
- The model predicts the class of the uploaded image (e.g., T-shirt, Trouser, Sneaker).

### Sample Images:
- The app displays a set of sample images for each clothing category.
- These sample images can be downloaded by clicking the "Download" button next to each image.

### Model:
- A Convolutional Neural Network (CNN) is used to classify the Fashion MNIST images into 10 classes.
- The model is trained using the PyTorch framework and is loaded into the app to make predictions.

---

## Sample Categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Fashion MNIST Dataset**: The Fashion MNIST dataset was created by Zalando Research.
- **Streamlit**: For providing a simple and powerful way to build interactive web applications.
- **PyTorch**: For its flexibility in building and training deep learning models.
