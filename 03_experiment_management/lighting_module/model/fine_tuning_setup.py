from lightning.pytorch.callbacks import BaseFinetuning

class ProgressiveBackboneFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch_1=5, unfreeze_at_epoch_2=10):
        """define specific epoch to start and stop fine tuning

        Args:
            unfreeze_at_epoch_1 (int, optional): epoch to unfreeze the last layer of the backbone. Defaults to 5.
            unfreeze_at_epoch_2 (int, optional): epoch to unfreeze the last three layers of the backbone. Defaults to 10.
        """
        super().__init__()
        self.unfreeze_at_epoch_1 = unfreeze_at_epoch_1
        self.unfreeze_at_epoch_2 = unfreeze_at_epoch_2

    def freeze_before_training(self, pl_module):
        """freeze all layers before training"""
        self.freeze(pl_module.model.features) 
        self.freeze(pl_module.model.avgpool)
        self.make_trainable(pl_module.model.classifier)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """ define procedure to unfreeze layers at specific epochs

        Args:
            pl_module (LightningModule): Lightning module
            current_epoch (int): Current epoch number
            optimizer (torch.optim.Optimizer): Optimizer instance
        """
        # === stage 1&2: unfreeze Backbone last layer (features[-1:]) and last 3 layers (features[-3:]) ===
        if current_epoch == self.unfreeze_at_epoch_1:
            self.make_trainable(pl_module.model.features[-1:])
        if current_epoch == self.unfreeze_at_epoch_2:
            self.make_trainable(pl_module.model.features[-3:])
