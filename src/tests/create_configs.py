"""
    Creation de fichier de config JSON et SLURM avec de nouveaux argumentes d'entrées.

"""
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def create_configs_from_args(
    num_nodes: int,
    num_gpus: int,
    num_cpus: int,
    outputs_folder: str,
    folder_templates:str=None,
    ):

    outputs_folder = Path(outputs_folder)
    assert outputs_folder.exists()

    if folder_templates is None:
        folder_templates = Path(__file__).parent.resolve() / "templates"

    # Load templates and create config files
    template_loader = FileSystemLoader(folder_templates)
    template_env = Environment(loader=template_loader)

    json_template = template_env.get_template("template.json")
    slurm_template = template_env.get_template("template.slurm")
    
    job_name = f"benchmark_{num_nodes}n_{num_gpus}g_{num_cpus}w"

    filled_json_template = json_template.render(
        job_name=job_name,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
    )

    filled_slurm_template = slurm_template.render(
        job_name=job_name,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
    )

    out_file_json = outputs_folder / f"{job_name}.json"
    out_file_slurm = outputs_folder / f"{job_name}.slurm"

    with open(out_file_json, "w") as json_file:
        json_file.write(filled_json_template)

    with open(out_file_slurm, "w") as slurm_file:
        slurm_file.write(filled_slurm_template)


def create_configs(
    outputs_folder: str,
    folder_templates:str=None,
    ):
    num_cpus = 10
    for num_nodes in [1, 2, 4]:
        for num_gpus in [1, 2, 4]:
            create_configs_from_args(
                num_nodes=num_nodes,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                outputs_folder=outputs_folder,
                folder_templates=folder_templates,
            )


if __name__ == '__main__':
    outputs_folder = "/home/dl/speillet/lightning/ensembling/benchmark/created_configs"
    create_configs(
        outputs_folder=outputs_folder,
    )
