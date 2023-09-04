import albumentations as A
import torch
import torch.nn as nn

from py_module.model import SMP_Unet_meta
from py_module.transforms import (
    Compose,
    CHW_to_HWC,
    HWC_to_CHW,
    ScaleImageToFloat,
    ToTorchTensor,
)


class Baseline():

    def __init__(self, config) -> None:
        self.config = config
        self.setup()

    def setup(self):
        self.model = self.set_model()
        self.collate_fn = self.set_collate_fn()
        self.criterion = self.set_criterion()
        self.step_fn = self.set_step_fn()
        self.predict_step_fn = self.set_predict_step_fn()
        self.transforms = self.set_transforms()

    # SETTERS
    def set_model(self):
        model = SMP_Unet_meta(
            n_channels=5,
            n_classes=self.config["num_classes"],
            use_metadata=self.config["use_metadata"],
        )
        return model


    def set_collate_fn(self):
        return None


    def set_transforms(self):
        if self.config["use_augmentation"] == True:
            transform_train = Compose([
                CHW_to_HWC(),
                A.VerticalFlip(p=0.5), 
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ScaleImageToFloat(), 
                HWC_to_CHW(), 
                ToTorchTensor(),
            ])
            transform_val = None
        else:
            transform_train = None
            transform_val = None

        return {
            "train": transform_train,
            "val": transform_val,
        }


    def set_criterion(self):
        if self.config["use_weights"] == True:
            with torch.no_grad():
                class_weights = torch.FloatTensor(self.config["class_weights"])
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion


    def set_step_fn(self):
        
        def step_fn(
                batch,
                object,
                use_metadata=self.config["use_metadata"],
            ):
            mtd = batch["mtd"] if use_metadata else ""
            logits = object.model(batch["img"], mtd)
            targets = batch["msk"]
            loss = object.criterion(logits, targets)
            return loss
        
        return step_fn


    def set_predict_step_fn(self):
        
        def predict_step_fn(
                batch,
                object,
                use_metadata=self.config["use_metadata"],
            ):
            mtd = batch["mtd"] if use_metadata else ""
            logits = object.model(batch["img"], mtd)
            proba = torch.softmax(logits, dim=1)
            batch["preds"] = torch.argmax(proba, dim=1)
            return batch
        
        return predict_step_fn


    def get_components(self):
        return vars(self)
