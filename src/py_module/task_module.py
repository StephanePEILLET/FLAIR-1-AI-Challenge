import pytorch_lightning as pl
import torch
from torchmetrics import MeanMetric


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        optimizer,
        scheduler=None,
        **kwargs,
    ):
        super().__init__()
        self.setup_components(kwargs)
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler


    def setup_components(self, components):
        for name, value in components.items():
            setattr(self, name, value)


    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss = None
            self.val_loss = MeanMetric()


    def step(self, batch):
        return self.step_fn(
            object=self,
            batch=batch,
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.train_loss.update(loss)
        return loss

    def training_epoch_end(self, outputs):
        self.train_epoch_loss = self.train_loss.compute()
        self.log(
            "train_loss",
            self.train_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.val_loss.update(loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.val_epoch_loss = self.val_loss.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.val_loss.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.predict_step_fn(
            object=self, 
            batch=batch
        )

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
                "name": "Scheduler",
            }
            config = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config
        else:
            return self.optimizer
