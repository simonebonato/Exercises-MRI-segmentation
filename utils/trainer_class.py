import numpy as np
import os
import torch
import torch.nn as nn
from typing import Optional


class Trainer(nn.Module):
    """
    Class for training a model.

    Args:
        train_data_loader (torch.utils.data.DataLoader): data loader for the training set
        val_data_loader (torch.utils.data.DataLoader): data loader for the validation set
        model (nn.Module): model to train
        criterion (nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epochs (int): number of epochs
        device (torch.device): device to use for the training
        early_stopping (int, optional): early stopping. -1 if you don't want to use Early Stopping, else choose an integer number that will represent the patience. Defaults to -1.
        train_from_checkpoint (Optional[str], optional): train from a checkpoint. Insert the path to the weights you wish to load. Defaults to None.
        fine_tune (bool, optional): fine tune the model, training only the last layer and freezing the others. Defaults to False.
        best_models_dir (str, optional): directory where the best models will be saved. Defaults to "best_models".
        mixed_precision (bool, optional): use mixed precision for the training. Defaults to False.
        Nit (Optional[int], optional): number of iterations for the training. Defaults to None.
        random_seed (int, optional): random seed for the training. Defaults to 42.
    """

    def __init__(
        self,
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
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.Nit = Nit
        self.random_seed = random_seed
        self.train_from_checkpoint = train_from_checkpoint
        self.fine_tune = fine_tune
        self.mixed_precision = mixed_precision
        self.best_models_dir = best_models_dir

        self.training_setup()

    def print_trainer_summary(self) -> None:
        """
        Prints a summary of the trainer parameters.
        """
        print("Trainer summary:")
        print(f"  -Model: {self.model._get_name()}")
        print(f"  -Loss function: {self.criterion._get_name()}")
        print(f"  -Optimizer: {str(self.optimizer)}")
        print(f"  -Device: {self.device}")
        print(f"  -Epochs: {self.epochs}")
        print(f"  -Early stopping: {self.early_stopping}")
        print(f"  -Train from checkpoint: {self.train_from_checkpoint}")
        print(f"  -Fine tune: {self.fine_tune}")
        print(f"  -Best models dir: {self.best_models_dir}")
        print(f"  -Mixed precision: {self.mixed_precision}")
        print(f"  -Nit: {self.Nit}")
        print(f"  -Random seed: {self.random_seed}\n")

    def training_setup(self) -> None:
        """
        Defines the model and other variables for the training.

        The following steps are executed:
        - set the random seed
        - load the model from a checkpoint if specified
        - set the requires_grad attribute to False for all the layers except the last two, for fine-tuning
        - set up the scaler for mixed precision training, if required
        - prints the number of iterations per epoch if Nit is set
        - create the directory for saving the best models if it doesn't exist

        """
        # set the random seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # load the model from a checkpoint if specified
        if self.train_from_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.train_from_checkpoint))
            print(f"\nModel loaded from {self.train_from_checkpoint}...\n")

        if self.fine_tune:
            # set the requires_grad attribute to False for all the layers except the last two
            # since they correspond to the last convolutional layer and its bias
            print("Fine tuning the model, training only last layer...\n")
            for param in list(self.model.parameters())[:-2]:
                param.requires_grad = False

        # setting up the scaler for mixed precision training, if required
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # prints the number of iterations per epoch if Nit is set
        if self.Nit is not None:
            print(
                f"Nit is set to {self.Nit}, training will stop after {self.Nit} iterations each epoch...\n"
            )

        # create the directory for saving the best models if it doesn't exist
        if not os.path.exists(self.best_models_dir):
            print(
                f"Creating directory {self.best_models_dir} for saving the best models...\n"
            )
            os.makedirs(self.best_models_dir)

        self.print_trainer_summary()

    def forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model and returns the loss, given a batch of data as input.

        Parameters
        :arg batch: a dictionary containing the data and labels of the batch

        Returns
        :return loss: the loss of the batch
        """
        x = batch["image"]["data"].to(self.device)
        y = batch["label"]["data"].to(self.device)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    def backward_pass(self, loss: torch.Tensor) -> None:
        """
        Performs a backward pass through the model and updates the parameters, given a loss as input.
        The backward pass is performed using mixed precision if specified in the constructor.

        Parameters
        :arg loss: the loss of the batch
        """
        # sets the gradients of all optimized torch.
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def training_loop(self):
        """
        Performs the training loop based on the parameters specified in the constructor.

        The following steps are executed:
        - set the training and validation loss to infinity
        - set the best validation loss to infinity
        - set the patience counter to 0
        - move the model to the device
        - for each epoch:
            - set the model to training mode
            - for each batch:
                - stop the training for this epoch if Nit is specified and reached
                - perform a forward pass and compute the loss
                - perform a backward pass and update the parameters
            - set the model to evaluation mode
            - for each batch:
                - perform a forward pass and compute the loss
            - compute the average loss for the epoch
            - compute the average validation loss for the epoch
            - print the training and validation loss for the epoch
            - save the model if the validation loss is the best so far
            - check if the validation loss has improved, if not, increase the patience counter
            - check if the patience counter has reached the early stopping threshold, if so, stop the training, otherwise reset the patience counter
        - save the last model
        """
        # variables setting for training
        loss = np.inf
        val_loss = np.inf
        self.best_val_loss = np.inf
        patience_counter = 0  # counter for early stopping

        self.model.to(self.device)

        print("Starting training...\n")
        for epoch in range(self.epochs):
            # training loop
            self.model.train()
            for idx, batch in enumerate(self.train_data_loader):
                # stop the training for this epoch if Nit is specified and reached
                if self.Nit is not None and idx == self.Nit:
                    break
                # check if mixed precision is required
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.mixed_precision, cache_enabled=False):  # type: ignore
                    loss = self.forward_pass(batch)
                self.backward_pass(loss)
            # validation loop
            self.model.eval()
            # no need to compute the gradients for the validation loop
            with torch.no_grad():
                for batch in self.val_data_loader:
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.mixed_precision, cache_enabled=False):  # type: ignore
                        val_loss = self.forward_pass(batch)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}"
            )

            # early stopping and saving the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0  # set patience counter to 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.best_models_dir, f"best_model.pth"),
                )

            elif self.early_stopping is not None:
                patience_counter += 1  # increase patience counter
                if (
                    patience_counter == self.early_stopping
                ):  # check if patience is reached
                    print("\nTraining stopped due to early stopping")
                    break

        # save the last model
        print(
            f"\nBest validation loss: {self.best_val_loss:.4f}\nSaving the last model..."
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.best_models_dir, f"last_model.pth"),
        )
