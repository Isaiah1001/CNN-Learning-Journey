# =========================
# 1) Imports
# =========================
# libraries

import os
import time
from scipy.io import loadmat
import copy
import matplotlib.pyplot as plt

import torch
from preprocess import get_mean_std, data_access, data_manipulate, get_dataloaders

# =========================
# 2) data loading and exploration
# =========================
# data location (for this project, the data is retrieved from 99_flower_data)
data_folder = '99_flower_data'
data_path = os.path.join(r'../', data_folder)

# check what's inside the data path
print(os.listdir(data_path))

# read in the text file containing the labels description
with open(os.path.join(data_path, "labels_description.txt"), "r", encoding="utf-8") as f:
    text_description = f.readlines()
print(text_description)

# read .mat file
data_labels_check = loadmat(os.path.join(data_path, 'imagelabels.mat'))
print('Keys in data dictionary:', data_labels_check.keys())
print('Data shape:', data_labels_check['labels'].shape)

# check availability of CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# =========================
# 3) Data Preparation and Processing
# =========================
# create dataset instance  
dataset = data_access(data_folder)

# plot image
def plot_image(ii):
    image, label, description = dataset[ii]
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f'image_{ii + 1:05d}: Label={label}, Description={description}')
    ax.axis('off')
    plt.show(block=False)  
    plt.pause(5)           
    plt.close(fig)       
plot_image(6316)

# Get all labels from the dataset object.
dataset_labels = dataset.data_labels
# Create a set of unique labels to remove duplicates.
unique_labels = set(dataset_labels)

# Iterate through each unique label.
for label in unique_labels:
    # Print the numerical label and its corresponding text description.
    print(f'Label: {label}, Description: {dataset.retrieve_description(label)}')
    
# calculate mean and std for normalization
mean, std = get_mean_std(data_path)
print(f'Calculated mean: {mean}, std: {std}')

# get transforms
basic, aug = data_manipulate(mean, std)
# to use the mean and std transforms for ImageNet please uncomment the following lines and comment out the above two lines
# basic, aug = data_manipulate(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# apply transforms and create dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    dataset=dataset,
    batch_size=32,
    train_transform=aug,
    val_transform=basic,
    train_size=0.2,
    val_size=0.15
)

# check dataloader details
def inspect_loader(loader, name):
    ds = loader.dataset
    print(f"\n=== {name} ===")
    print("num_batches:", len(loader))
    print("num_samples:", len(ds))
    print("dataset type:", type(ds))
    print("transform:", getattr(ds, "transform", None))

# inspect dataloaders
inspect_loader(train_loader, "Train")
inspect_loader(val_loader, "Validation")
inspect_loader(test_loader, "Test")

# plot a batch of images from the training loader
train_loader_trans= train_loader.dataset
def plot_image(ii):
    image, label, description = train_loader_trans[ii]

    # CHW -> HWC
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img = image.transpose(1, 2, 0)

    plt.figure()
    plt.imshow(img)
    plt.title(f'Label={label}, Description={description}')
    plt.axis('off')
    plt.show()

# plot the image in the training dataset
plot_image(200)

# =========================
# 4) model
# =========================
from model import SimpleCNN
from model import model_detail, para_debug, check_layer_parameters, check_total_parameters

# elect model architecture
simple_cnn = SimpleCNN()

# check model details
images, labels, description = next(iter(train_loader))
debug_model = model_detail()
debug_output = debug_model(images)

debug_model_v2 = para_debug()
debug_output_v2 = debug_model_v2(images)

check_layer_parameters(simple_cnn)
check_total_parameters(simple_cnn)

# =========================
# 5) training and evaluation
# =========================
# import training loop
from model import training_loop
# loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(simple_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
# train the model
trained_model, training_metrics = training_loop(
    model=simple_cnn,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    epochs=40
)

# =========================
# 6) Plot training and validation metrics
# =========================
def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
plot_training_metrics(training_metrics)
