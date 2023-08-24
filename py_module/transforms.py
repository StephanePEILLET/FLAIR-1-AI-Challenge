from typing import Any, Dict, Union

import numpy as np
import torch

class BasicTransform:
    def __init__(self):
        self.params: Dict[Any, Any] = {}
        self.img_only: bool = False
        self.mask_only: bool = False

    def __call__(
        self, image: np.ndarray = None, mask: np.ndarray = None
    ) -> Dict[str, Any]:
        if self.img_only:
            return {"image": self.apply_to_img(image), "mask": mask}
        elif self.mask_only:
            return {"image": image, "mask": self.apply_to_mask(mask)}
        else:
            return {"image": self.apply_to_img(image), "mask": self.apply_to_mask(mask)}

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Compose(object):
    """Compose function differs from torchvision Compose as sample argument is passed unpacked to match albumentation
    behaviour.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **sample):
        for t in self.transforms:
            sample = t(**sample)
        return sample


class ScaleImageToFloat(BasicTransform):
    """
    scale an input sample to float image between [0, 1]
    """

    def __init__(
        self,
        scale_factor: float = 255,
        clip: bool = False,
        img_only: bool = True,
        mask_only: bool = False,
    ):
        super(ScaleImageToFloat, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only
        self.scale_factor = scale_factor
        self.clip = clip

    def _scale_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.multiply(arr, 1.0 / self.scale_factor, dtype=np.float32)
        if self.clip:
            return np.clip(arr, 0, 1)
        else:
            return arr

    def apply_to_img(self, img: np.ndarray) -> np.ndarray:
        return self._scale_array(img)

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        return self._scale_array(mask)


class HWC_to_CHW(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(HWC_to_CHW, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(
        array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        array = array.swapaxes(0, 2).swapaxes(1, 2)
        return array

    def apply_to_img(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(img)

    def apply_to_mask(
        self, mask: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(mask)


class CHW_to_HWC(BasicTransform):
    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(CHW_to_HWC, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    @staticmethod
    def swap_axes(
        array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
        array = array.swapaxes(0, 2).swapaxes(0, 1)
        return array

    def apply_to_img(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(img)

    def apply_to_mask(
        self, mask: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.swap_axes(mask)


class ToTorchTensor(BasicTransform):
    """
    Convert (image, mask) sample from numpy array to torch tensor
    Note:
        Output type cast to torch.float32
    Args:
        img_only (str) : Apply transform only to image
        mask_only (List(Int)) : Apply transform only to mask
    """

    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(ToTorchTensor, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    def apply_to_img(self, img: np.ndarray):
        return torch.from_numpy(img.copy()).type(torch.float32)

    def apply_to_mask(self, mask: np.ndarray):
        return torch.from_numpy(mask.copy()).type(torch.float32)


class TensorToArray(BasicTransform):
    """
    Convert (image, mask) sample torch tensor to numpy array
    Note:
       tensor in gpu
    Args:
        img_only (str) : Apply transform only to image
        mask_only (List(Int)) : Apply transform only to mask
    """

    def __init__(self, img_only: bool = False, mask_only: bool = False):
        super(TensorToArray, self).__init__()
        self.img_only = img_only
        self.mask_only = mask_only

    def apply_to_img(self, img) -> np.ndarray:
        return img.cpu().numpy()

    def apply_to_mask(self, mask) -> np.ndarray:
        return mask.cpu().numpy()
