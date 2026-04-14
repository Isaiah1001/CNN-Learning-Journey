#model/training_loop.py
import time
import copy
import torch

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """training epoch definition

    Args:
        model: model to be trained
        dataloader: training dataloader
        loss_fn: loss function definition
        optimizer: optimizer definition
        device: cpu or gpu device

    Returns:
        ave_loss: average loss of the epoch
    """
    model.train()
    total_loss = 0
    for data, targets, _ in dataloader:
        data = data.to(device)
        targets = targets.to(device)
        # zero gradients
        optimizer.zero_grad()
        # forward
        scores = model(data)
        loss = loss_fn(scores, targets)

        # backward
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validation_epoch(model, dataloader, loss_fn, device):
    """validation step

    Args:
        model: training model
        dataloader: validation dataloader
        loss_fn: loss function definition
        device: cpu or gpu device

    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets, _ in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = loss_fn(scores, targets)
            total_loss += loss.item()
            _, preds = torch.max(scores, 1)
            correct += (preds == targets).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

def training_loop(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    """ loop for training and validation

    Args:
        model: model to be trained
        train_loader: training dataloader
        val_loader: validation dataloader
        loss_fn: loss function definition
        optimizer: optimizer definition
        device: cpu or gpu device
        epochs: number of epochs to train

    Returns:
        tuple: (trained model, training and validation metrics)
    """
    
    # move the model to the target device
    model.to(device)
    # store best validation accuracy
    best_val_accuracy = 0.0
    # store best model state
    best_model_state = None
    # store best epoch
    best_epoch = 0
    # store training and validation history
    train_losses, val_losses, val_accuracies = [], [], []
    
    print("--- Training Started ---")
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = validation_epoch(model, val_loader, loss_fn, device)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | lr={lr:.6f}")
        # print(f"  Val:   loss={val_loss:.4f}, acc={val_accuracy*100:.2f}%")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            
        print(f'Epoch {epoch+1}/{epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f"lr={lr:.6f}")
        print(f'\tTrain Loss: {train_loss:.4f}| Val. Loss: {val_loss:.4f}')
        print(f'Val. Acc: {val_accuracy*100:.2f}%')
    print("--- Finished Training ---")
    
    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy*100:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    return model, metrics