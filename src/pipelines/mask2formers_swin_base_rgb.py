import albumentations as A
import numpy as np
import torch
import torch.nn as nn

from py_module.model import SMP_Unet_meta
from py_module.transforms import (
    Compose,
    CHW_to_HWC,
    HWC_to_CHW,
    ToTorchTensor,
)

MEAN = np.array([0.44050665, 0.45704361, 0.42254708]) 
STD = np.array([0.20264351, 0.1782405 , 0.17575739]) 


# Model mask2formers_swin_base_rgb
class Mask2Formers_Swin_Base_RGB():

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

        from transformers import MaskFormerImageProcessor
        # Create a preprocessor
        self.preprocessor = MaskFormerImageProcessor(
            ignore_index=0, 
            reduce_labels=False, 
            do_resize=False, 
            do_rescale=False, 
            do_normalize=False
        )

    def set_model(self):
        from transformers import Mask2FormerForUniversalSegmentation     
        from py_module.constants import ALAN_ID2LABEL

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-IN21k-ade-semantic",
            id2label=ALAN_ID2LABEL,
            ignore_mismatched_sizes=True,
        )
        return model


    def set_collate_fn(self):

        def collate_fn(batch):
            """
                This function pads the inputs to the same size, and creates a pixel mask actually padding 
                isn't required here since we are cropping.
            """
            batch = self.preprocessor(
                images=batch["img"],
                segmentation_maps=batch["msk"],
                return_tensors="pt",
            )
            return batch
    
        return collate_fn


    def set_transforms(self):
        if self.config["use_augmentation"] == True:
            transform_train = Compose([
                CHW_to_HWC(),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(),
                A.RandomBrightnessContrast(),
                A.Normalize(mean=MEAN, std=STD),
                HWC_to_CHW(), 
                ToTorchTensor(),
            ])
            transform_val= A.Compose([
                A.Normalize(mean=MEAN, std=STD),
                HWC_to_CHW(), 
                ToTorchTensor(),
            ])
        else:
            transform_train = None
            transform_val = None

        return {
            "train": transform_train,
            "val": transform_val,
        }


    def set_criterion(self):
        return None


    def set_step_fn(self):

        def step_fn(
                model, 
                batch,
                criterion,
            ):

            outputs = model(
                pixel_values=batch["pixel_values"],
                mask_labels=batch["mask_labels"],
                class_labels=batch["class_labels"],
            )
            loss = outputs.loss
            return loss
        
        return step_fn


    def set_predict_step_fn(self):

        def predict_step_fn(
                model, 
                batch, 
                preprocessor,
            ):
            outputs = model(pixel_values=batch["pixel_values"])
            target_sizes = [(image.shape[0], image.shape[1]) for image in batch["img"]]
            batch["preds"] = preprocessor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes,
            )
            return batch
        
        return predict_step_fn


    def get_components(self):
        return vars(self)