import os

import numpy as np
import rasterio
import torch
from skimage import img_as_float
from torch.utils.data import Dataset

from py_module.transforms import (
    Compose,
    CHW_to_HWC,
    HWC_to_CHW,
    ScaleImageToFloat,
    ToTorchTensor,
)

class Fit_Dataset(Dataset):
    def __init__(
        self,
        dict_files,
        num_classes=13,
        use_metadata=True,
        transforms=None,
    ):
        self.list_imgs = np.array(dict_files["IMG"])
        self.list_msks = np.array(dict_files["MSK"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD"])
        if transforms is None:
            self.transforms = Compose([
                ScaleImageToFloat(), 
                ToTorchTensor(),
            ])
        else:
            self.transforms = transforms

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            array[array > self.num_classes] = self.num_classes
            array = array - 1
            array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        img = self.read_img(raster_file=self.list_imgs[index])
        msk = self.read_msk(raster_file=self.list_msks[index])

        sample = {"image": img, "mask": msk}
        transformed_sample = self.transforms(**sample)
        batch = {
            "img": transformed_sample["image"], 
            "msk": transformed_sample["mask"],
        }

        if self.use_metadata == True:
            mtd = torch.as_tensor(self.list_metadata[index], dtype=torch.float)
            batch["mtd"] = mtd

        return batch


class Predict_Dataset(Dataset):
    def __init__(
            self, 
            dict_files, 
            num_classes=13, 
            use_metadata=True,
            transforms=None,    
        ):
        self.list_imgs = np.array(dict_files["IMG"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata

        if use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD"])
        
        if transforms is None:
            self.transforms = Compose([
                ScaleImageToFloat(img_only=True), 
                ToTorchTensor(img_only=True),
            ])
        else:
            self.transforms = transforms

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)
        sample = {"image": img}
        transformed_sample = self.transforms(**sample)
        batch = {
            "img": transformed_sample["image"], 
             "id": "/".join(image_file.split("/")[-4:]),
        }
        if self.use_metadata == True:
            mtd = torch.as_tensor(self.list_metadata[index], dtype=torch.float)
            batch["mtd"] = mtd
        return batch
