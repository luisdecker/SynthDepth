from pytorch_lightning import Callback


class UnfreezeEncoder(Callback):
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch == self.unfreeze_epoch:
            # Unfreeze encoder parameters
            for param in pl_module.encoder.parameters():
                param.requires_grad = True
            print("Encoder unfrozen at epoch", current_epoch)
