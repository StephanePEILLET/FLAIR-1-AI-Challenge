"""
    Test du code en mono-gpu puis en multi-gpus.
"""

import os
from pathlib import Path
from pipelines.pipeline_factory import PIPELINE_FACTORY

import shutil
import tempfile
import yaml


def manage_outputs(out_folder=None):
    """
        Gestion du répétoire de sortie.
    """
    if out_folder is None:
        current_pos = Path(__file__).parent.resolve()
        out_folder = current_pos / "outputs"
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder


def launch_local_tests():
    """
        Lance des tests en local.
    """
    out_folder = manage_outputs()

    local_config = {
        #### DATA PATHS
        "data" : {
            "path_aerial_train": "/home/dl/speillet/ocsge/flair1/data/subset/train",
            "path_aerial_test": "/home/dl/speillet/ocsge/flair1/data/subset/flair_1_aerial_test",
            "path_labels_train": "/home/dl/speillet/ocsge/flair1/data/subset/train",
            "path_labels_test": "/home/dl/speillet/ocsge/flair1/data/subset/flair_1_labels_test",
            "path_metadata_aerial": "/home/dl/speillet/ocsge/flair1/FLAIR-1-AI-Challenge/data/flair-1_metadata_aerial.json",
        },
        "outputs": {
            "out_model_name": "FLAIR_debug.ckpt",
        },
        "num_classes": 13,

        #### TRAINING CONF
        "use_weights": True,
        "class_weights": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
        "use_metadata": False,
        "use_augmentation": True,

        #### HYPERPARAMETERS
        "batch_size": 2,
        "learning_rate": 0.02,
        "num_epochs": 2,

        #### COMPUTATIONAL RESSOURCES
        "accelerator": "gpu", # or cpu
        "num_nodes": 1,
        "gpus_per_node": 1,
        "strategy": None, # null if only one GPU, else 'ddp'
        "num_workers": 8,

        #### PRINT PROGRESS
        "enable_progress_bar": True,
        "progress_rate": 10,
    }

    # list_pipelines = list(PIPELINE_FACTORY.keys())
    list_pipelines = ["baseline"]
    for pipeline in list_pipelines:

        data = local_config.copy()
        data["pipeline"] = pipeline

        pipeline_outputs = out_folder / pipeline
        local_config["outputs"]["out_folder"] = pipeline_outputs.as_posix()
        yaml_tempfile = tempfile.NamedTemporaryFile(suffix=".yml", delete=True)
        try:
            yaml_filename = yaml_tempfile.name
            with open(yaml_filename, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)
            os.system(f"python src/main.py --config_file {yaml_filename}")
        finally:
            yaml_tempfile.close()

        if pipeline_outputs.exists():
            shutil.rmtree(pipeline_outputs.as_posix(), ignore_errors=True)


def launch_slurm_tests():
    """
        Lance des tests sur un cluster utilisant SLURM.
    """
    cluster_config = {
        #### DATA PATHS
        "data" : {
            "path_aerial_train": "/home/dl/speillet/ocsge/flair1/data/subset/train",
            "path_aerial_test": "/home/dl/speillet/ocsge/flair1/data/subset/flair_1_aerial_test",
            "path_labels_train": "/home/dl/speillet/ocsge/flair1/data/subset/train",
            "path_labels_test": "/home/dl/speillet/ocsge/flair1/data/subset/flair_1_labels_test",
            "path_metadata_aerial": "/home/dl/speillet/ocsge/flair1/FLAIR-1-AI-Challenge/data/flair-1_metadata_aerial.json",
        },
        "outputs": {
            "out_model_name": "FLAIR_debug.ckpt",
        },
        "num_classes": 13,

        #### TRAINING CONF
        "use_weights": True,
        "class_weights": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
        "use_metadata": False,
        "use_augmentation": True,

        #### HYPERPARAMETERS
        "batch_size": 2,
        "learning_rate": 0.02,
        "num_epochs": 2,

        #### COMPUTATIONAL RESSOURCES
        "accelerator": "gpu", # or cpu
        "num_nodes": 1,
        "gpus_per_node": 1,
        "strategy": None, # null if only one GPU, else 'ddp'
        "num_workers": 8,

        #### PRINT PROGRESS
        "enable_progress_bar": True,
        "progress_rate": 10,
    }

    out_folder = manage_outputs()
    list_pipelines = list(PIPELINE_FACTORY.keys())
    list_pipelines = ["baseline"]

    for pipeline in list_pipelines:

        data = cluster_config.copy()
        data["pipeline"] = pipeline

        pipeline_outputs = out_folder / pipeline
        cluster_config["outputs"]["out_folder"] = pipeline_outputs.as_posix()
        # Creation du fichier .yml
        yaml_tempfile = tempfile.NamedTemporaryFile(suffix=".yml", delete=True)
        try:
            yaml_filename = yaml_tempfile.name
            with open(yaml_filename, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)
            
            # Une fois le fichier .yml créer
            
            # os.system(f"python src/main.py --config_file {yaml_filename}")
        finally:
            yaml_tempfile.close()


        # Supression des fichires de configs créés

        if pipeline_outputs.exists():
            shutil.rmtree(pipeline_outputs.as_posix(), ignore_errors=True)