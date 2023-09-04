"""
    Script permettant de créer via des liens symboliques un ensemble de répertoires contenant les données FLAIR
    selon l'organisation attendue par le code des 2nd participants au challenge FLAIR.
"""
import os
from pathlib import Path


def create_symbolic_dataset(
        train_dataset: str,
        test_aerial: str,
        test_labels: str,
        data_folder: str = None,
        verbose: bool = False,
        
    ):
    """
        Code pour créer rapidement un répertoire data contenant que des liens symboliques 
        pointant vers les données réelles.
    """
    train_dataset = Path(train_dataset)
    test_aerial = Path(test_aerial)
    test_labels = Path(test_labels)
    data_folder = Path(data_folder)

    list_train_imgs = sorted(list(train_dataset.glob("*/*/*/img/*.tif")))
    list_train_msks = sorted(list(train_dataset.glob("*/*/*/msk/*.tif")))
    assert len(list_train_imgs) == len(list_train_msks)

    percent = 0.9
    idx_split = round(len(list_train_imgs) * percent)

    train_imgs = list_train_imgs[:idx_split]
    train_msks = list_train_msks[:idx_split]

    val_imgs = list_train_imgs[idx_split:]
    val_msks = list_train_msks[idx_split:]

    test_imgs = sorted(list(test_aerial.glob("*/*/img/*.tif")))
    test_msks = sorted(list(test_labels.glob("*/*/msk/*.tif")))
    assert len(test_imgs) == len(test_msks)

    if data_folder is None:
        target_folder = Path('data')
    else:
        target_folder = Path(data_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for split_name, imgs, msks in zip(
            ["train", "val", "test"],
            [train_imgs, val_imgs, test_imgs],
            [train_msks, val_msks, test_msks],
    ):
        if verbose:
            print(60 * "#")
            print(f"{split_name} => ", "Nbr imgs: ", len(imgs), "/  Nbr msks: ", len(msks))

        split_folder = target_folder/ split_name
        split_folder.mkdir(parents=True, exist_ok=True)

        for img, msk in zip(imgs, msks):
            # Create symbolic link for each sample
            (split_folder / img.name).symlink_to(img)
            (split_folder / msk.name).symlink_to(msk)


def main():

    train_dataset = "/home/dl/speillet/ocsge/flair1/data/data_flair-one/flair-one_train"
    test_aerial = "/home/dl/speillet/ocsge/flair1/data/data_flair-one/flair_1_aerial_test"
    test_labels = "/home/dl/speillet/ocsge/flair1/data/data_flair-one/flair_1_labels_test"
    data_folder = "/home/dl/speillet/ocsge/flair1/FLAIR-1-AI-Challenge/data"

    create_symbolic_dataset(
        train_dataset=train_dataset,
        test_aerial=test_aerial,
        test_labels=test_labels,
        data_folder=data_folder,
        verbose=False,
    )


if __name__ == "__main__":
    main()