import numpy as np
import os
import torch
import torch.nn as nn
from typing import Optional


def forward_pass(model: nn.Module, batch, criterion, device):
    x, y = batch["image"]["data"].to(device), batch["label"]["data"].to(device)
    logits = model(x)
    loss = criterion(logits, y)
    return loss


def backward_pass(loss, optimizer, scaler=None):
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def training_setup(
    model: nn.Module,
    train_from_checkpoint: Optional[str] = None,
    fine_tune: bool = False,
    best_models_dir: str = "best_models",
    mixed_precision: bool = False,
    Nit: Optional[int] = None,
):
    # load the model from a checkpoint if specified
    if train_from_checkpoint is not None:
        model.load_state_dict(torch.load(train_from_checkpoint))
        print(f"\nModel loaded from {train_from_checkpoint}...\n")

    if fine_tune:
        # set the requires_grad attribute to False for all the layers except the last two
        # since they correspond to the last convolutional layer and its bias
        print("Fine tuning the model, training only last layer...\n")
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False

    # setting up the scaler for mixed precision training, if required
    if mixed_precision:
        print("Using mixed precision training...\n")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if Nit is not None:
        print(
            f"Nit is set to {Nit}, training will stop after {Nit} iterations each epoch...\n"
        )

    # create the directory for saving the best models if it doesn't exist
    if not os.path.exists(best_models_dir):
        print(f"Creating directory {best_models_dir} for saving the best models...\n")
        os.makedirs(best_models_dir)

    return model, scaler


def training_loop(
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    early_stopping: int = -1,
    train_from_checkpoint: Optional[str] = None,
    fine_tune: bool = False,
    best_models_dir: str = "best_models",
    mixed_precision: bool = False,
    Nit: Optional[int] = None,
):
    model, scaler = training_setup(
        model, train_from_checkpoint, fine_tune, best_models_dir, mixed_precision, Nit
    )

    # variables setting for training
    loss = np.inf
    val_loss = np.inf
    best_val_loss = np.inf
    patience_counter = 0  # counter for early stopping

    model.to(device)

    for epoch in range(epochs):
        # training loop
        model.train()
        for idx, batch in enumerate(train_data_loader):
            # stop the training for this epoch if Nit is specified and reached
            if idx is not None and idx == Nit:
                break
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = forward_pass(model, batch, criterion, device)
                backward_pass(loss, optimizer, scaler)
            else:
                loss = forward_pass(model, batch, criterion, device)
                backward_pass(loss, optimizer)

        # validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_data_loader:
                val_loss = forward_pass(model, batch, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs}, train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}"
        )

        # early stopping and saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # set patience counter to 0
            torch.save(
                model.state_dict(), os.path.join(best_models_dir, f"best_model.pth")
            )

        elif early_stopping != -1:
            patience_counter += 1  # increase patience counter
            if patience_counter == early_stopping:  # check if patience is reached
                print("\nTraining stopped due to early stopping")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}\nSaving the last model...")
    torch.save(model.state_dict(), os.path.join(best_models_dir, f"last_model.pth"))
