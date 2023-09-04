#!/bin/bash

#SBATCH --account=tel@v100                    # sur quel projet imputer
#SBATCH --job-name=slurm_1n_1g_8workers       # nom du job
#SBATCH --nodes=1                             # on demande 1 noeuds
#SBATCH --ntasks-per-node=1                   # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                          # nombre de GPU
#SBATCH --cpus-per-task=8                     # nombre de coeurs CPU par tache
#SBATCH --time=00:30:00                       # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --hint=nomultithread

#SBATCH --output=/gpfsssd/worksf/projects/rech/tel/uug84ql/logs/slurm_1n_1g_8workers_%j.out      # nom du fichier de sortie
#SBATCH --error=/gpfsssd/worksf/projects/rech/tel/uug84ql/logs/slurm_1n_1g_8workers_%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load python

conda deactivate
conda activate odeon
set -x

export TORCH_HOME=$HOME/.cache/torch/
export TMPDIR=$SCRATCH/torch_tmp

# execution du code
srun python /gpfsssd/worksf/projects/rech/tel/uug84ql/FLAIR-1-AI-Challenge/src/main.py --config_file /gpfsssd/worksf/projects/rech/tel/uug84ql/FLAIR-1-AI-Challenge/configs/slurm_1n_1g_8workers.yml
