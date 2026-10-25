#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -J sweep_early_fusion
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o sweep_early_fusion_%J.out
#BSUB -e sweep_early_fusion_%J.err

source venv/bin/activate

export DATASET_VARIANT=original
export DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/ufc10/}"

wandb agent magnus3400-danmarks-tekniske-universitet-dtu/idlcv-project2/kemnoqhk
