import torch
import torchvision.models as tv_models
from torchmetrics import Accuracy
import lightning.pytorch as pl


class FlowerLightModule(pl.LightningModule):
    def __init__(self, learning_rate= 1e-2, momentum =0.9, weight_decay=1e-4, num_classes=102):
        """initialize the model module

        Args:
            model: the neural network model
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            weight_decay (float, optional): weight decay for optimizer. Defaults to 0.0001.
            device (str, optional): device to run the model on. Defaults to None (will use 'cuda' if available).
        """
        super().__init__()
        # Save the hyperparameters passed to the constructor. This makes them
        # accessible via `self.hparams` and logs them automatically.
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        # Import the pre-trained EfficientNet-B0 model and modify the classifier head.
        self.model = tv_models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features
        new_fc_layer = torch.nn.Linear(in_features, self.num_classes)
        self.model.classifier[1] = new_fc_layer
        # Initialize the loss function.
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initialize metrics to track accuracy for training and validation.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor containing a batch of images.

        Returns:
            The output tensor (logits) from the model.
        """
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx = None):
        inputs, labels, _ = batch
        outputs = self(inputs)
        # update and log training loss 
        loss = self.loss_fn(outputs, labels)
        bs = inputs.size(0)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        # update and log training accuracy
        self.train_accuracy.update(outputs, labels)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        inputs, labels, _ = batch
        outputs = self(inputs)
        # update and log validation loss
        loss = self.loss_fn(outputs, labels)
        bs = inputs.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        # update and log validation accuracy
        self.val_accuracy.update(outputs, labels)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        
    def configure_optimizers(self):
        # backbone_params = list(self.model.features.parameters())
        head_params = list(self.model.classifier.parameters())
        optimizer = torch.optim.SGD(
            [
                # {"params": backbone_params, "lr": 1e-3},   # backbone small lr
                {"params": head_params, "lr": 1e-1},       # head large lr
            ],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=1e-4
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
