import torch
from torch import nn

import torchvision as tv
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timeit import default_timer as timer

from torch.utils.data import DataLoader

# Define the custom transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor values
])

# Load the FashionMNIST dataset with the custom transformation
train_data = datasets.FashionMNIST(
    root="data",  # Location to download the dataset
    train=True,  # Use the training dataset
    download=True,  # Download if not already present
    transform=transform,  # Apply the custom transform
    target_transform=None  # No transformation applied to the labels
)

test_data = datasets.FashionMNIST(
    root="data",  # Location to download the dataset
    train=False,  # Use the test dataset
    download=True,  # Download if not already present
    transform=transform,  # Apply the custom transform
    target_transform=None  # No transformation applied to the labels
)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


class_dict= train_data.class_to_idx
class_list = train_data.classes

BATCH_SIZE= 32
train_dataloader= DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True )
test_dataloader= DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False )
print (f"Original size / batched data sets : {len(train_data)}/{len(train_dataloader)} so batch size is {len(train_data)/len(train_dataloader)} \nSimilarly training data is {len(test_data)}/{len(test_dataloader)} : {len(test_data)/len(test_dataloader)}")


### Loss, Optimizer and Acc function

loss_fn = nn.CrossEntropyLoss().to(device)

def acc_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


from tqdm.auto import tqdm

torch.manual_seed(42)
def train_model(epochs:int, model:torch.nn.Module):
  opp_fn = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_fn = nn.CrossEntropyLoss().to(device)
  model.to(device)  # Ensure model is on the correct device
  start_time = timer()
  epochs = epochs

  for epoch in tqdm(range(epochs)):
      print(f"Epoch: {epoch + 1}/{epochs}\n---")
      train_loss, train_acc = 0, 0

      # Training loop
      for batch, (X, y) in enumerate(train_dataloader):
          X, y = X.to(device), y.to(device)  # Move to device
          model.train()

          y_pred = model(X)  # X is already on the device
          loss = loss_fn(y_pred, y)  # Ensure loss_fn is compatible
          train_loss += loss.item()  # Use .item() for scalar tensors
          train_acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
          opp_fn.zero_grad()
          loss.backward()
          opp_fn.step()

          if batch % 400 == 0:
              print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

      train_loss /= len(train_dataloader)
      train_acc /= len(train_dataloader)
      print(f"Avg Train Loss: {train_loss:.4f} || Avg Train Acc: {train_acc:.4f}")

      # Evaluation loop
      test_loss, test_acc = 0, 0
      model.eval()
      with torch.inference_mode():
          for X_test, y_test in test_dataloader:
              X_test, y_test = X_test.to(device), y_test.to(device)
              test_pred = model(X_test)
              test_loss += loss_fn(test_pred, y_test).item()
              test_acc += acc_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

      test_loss /= len(test_dataloader)
      test_acc /= len(test_dataloader)
      print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")

  end_timer = timer()
  print(f"Training complete in {end_timer - start_time:.2f} seconds.")


"""
  Creating a tinyVGG from CNN github website
"""

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1= nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2= nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        #print(x.shape)
        x=self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.classifier(x)
        return x


# Initialize the model
model_1 = FashionMNISTModelV1(input_shape=1, #1 color channel
                              hidden_units=64,
                              output_shape=len(class_list)).to(device)
print(model_1)




from pathlib import Path

MODEL_PATH= Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME= "O1_fashionMNIST_model.pth"
MODEL_SAVE_PATH= MODEL_PATH/MODEL_NAME


try:
    model_1.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    print("Model weights loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model weights: {e}")
    exit()

train_model(2,model_1)

torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

