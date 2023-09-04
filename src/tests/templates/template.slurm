#!/bin/bash
#SBATCH --job-name={{job_name}}       # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --nodes={{num_nodes}}                    # on demande 2 noeuds
#SBATCH --ntasks-per-node={{num_gpus}}          # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:{{num_gpus}}                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task={{num_cpus}}           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=logs/{{job_name}}_%j.out      # nom du fichier de sortie
#SBATCH --error=logs/{{job_name}}_%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=tel@v100 # besoin de définir le compte projet sur lequel on veut consommer de la ressource.

module purge

module load python

conda deactivate
conda activate odeon

set -x
export PYTHONPATH=$WORK/odeon-landcover
export TORCH_HOME=$HOME/.cache/torch
export TMPDIR=$SCRATCH/torch_tmp

srun python $WORK/odeon-landcover/odeon/main.py train -c /gpfswork/rech/ofp/uug84ql/benchmark/configs/{{job_name}}.json
