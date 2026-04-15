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
from model import training_loop, count_parameters, save_checkpoint, load_checkpoint, build_efficientnet_b0, unfreeze_last_3block_and_head


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
    # 4) model: load checkpoint and unfreeze the last block + head
    # =========================
    checkpoint_path = os.path.join('checkpoints', 'efficientnet_b0_flower.pth')
    fine_tune_epochs = 40

    base_model = build_efficientnet_b0(num_classes=102, weights=None)
    loaded_epoch, previous_metrics = load_checkpoint(
        checkpoint_path,
        base_model,
        optimizer=None,
        device=device
    )
    print(f"Resume from checkpoint epoch: {loaded_epoch}")

    unfreeze_last_3block_and_head(base_model)
    print("Model loaded for stage-2 fine-tuning:", base_model)

    # check the number of trainable parameters and total parameters
    total_params, trainable_params = count_parameters(base_model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # =========================
    # 5) training and evaluation
    # =========================
    # Rebuild optimizer because the trainable parameter set changed.
    loss_fn = torch.nn.CrossEntropyLoss() #label_smoothing=0.1
    optimizer = torch.optim.SGD([
        {'params': base_model.features[-3].parameters(), 'lr':1e-4},
        {'params': base_model.features[-2].parameters(), 'lr':5e-4},
        {'params': base_model.features[-1].parameters(), 'lr':1e-3},
        {'params': base_model.classifier.parameters(), 'lr':1e-2},
        ], momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, fine_tune_epochs, eta_min=5e-5)

    trained_model, training_metrics = training_loop(
        model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=fine_tune_epochs,
        # scheduler=scheduler
    )

    merged_metrics = [
        previous_metrics[0] + training_metrics[0],
        previous_metrics[1] + training_metrics[1],
        previous_metrics[2] + training_metrics[2],
    ]

    # =========================
    # 6) Save stage-2 checkpoint
    # =========================
    finetune_checkpoint_path = os.path.join('checkpoints', 'efficientnet_b0_flower_last3layer_plus_head_difflr.pth')
    save_checkpoint(
        trained_model,
        optimizer,
        epoch=loaded_epoch + fine_tune_epochs,
        metrics=merged_metrics,
        filepath=finetune_checkpoint_path
    )

    # =========================
    # 7) Plot training and validation metrics
    # =========================
    plot_training_metrics(merged_metrics)