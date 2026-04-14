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
from postprocess import plot_image, plot_image_LD, plot_training_metrics
from model import training_loop, count_parameters, save_checkpoint, load_checkpoint
if __name__ == '__main__':
    # =========================
    # 2) data loading and exploration
    # =========================
    torch.manual_seed(42) # Set a fixed random seed for reproducibility

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

    plot_image(6316, dataset)

    # Get all labels from the dataset object.
    dataset_labels = dataset.data_labels
    # Create a set of unique labels to remove duplicates.
    unique_labels = set(dataset_labels)

    # Iterate through each unique label.
    for label in unique_labels:
        # Print the numerical label and its corresponding text description.
        print(f'Label: {label}, Description: {dataset.retrieve_description(label)}')
        
    # calculate mean and std for normalization
    # mean, std = get_mean_std(data_path)
    # to use the mean and std transforms calculated by this dataset，comment the following lines and uncomment above two lines
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225] # using the mean and std from ImageNet dataset
    # get transforms
    basic, aug = data_manipulate(mean, std)

    print(f'Calculated mean: {mean}, std: {std}')
    # apply transforms and create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=128,
        train_transform=aug,
        val_transform=basic,
        train_size=0.8,
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
    plot_image_LD(200, train_loader_trans)

    # =========================
    # 4) model
    # =========================
    import torchvision.models as tv_models
    base_model = tv_models.efficientnet_b0(weights='IMAGENET1K_V1')
    print(base_model)
    for param in base_model.parameters():
        param.requires_grad = False  # freeze the base model
        
    # modify the classifier head
    original_fc = base_model.classifier
    print("Model's Original Fully Connected Layer:")
    print(original_fc)
    num_features = original_fc[1].in_features
    num_headers = 102  # number of classes in the flower dataset
    new_fc_layer = torch.nn.Linear(num_features, num_headers)
    base_model.classifier[1] = new_fc_layer
    print("Model's New Fully Connected Layer:",base_model.classifier)
    print("Model's Updated Structure:", base_model)

    # check the number of trainable parameters and total parameters
    total_params, trainable_params = count_parameters(base_model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # =========================
    # 5) training and evaluation
    # =========================
    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-1, weight_decay=1e-4)
    # train the model
    trained_model, training_metrics = training_loop(
        model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=40
    )

    # =========================
    # 6) Save checkpoint
    # =========================
    checkpoint_path = os.path.join('checkpoints', 'efficientnet_b0_flower.pth')
    save_checkpoint(trained_model, optimizer, epoch=40, metrics=training_metrics, filepath=checkpoint_path)

    # =========================
    # 7) Plot training and validation metrics
    # =========================
    plot_training_metrics(training_metrics)
