#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -J train_early_fusion
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o train_early_fusion_%J.out
#BSUB -e train_early_fusion_%J.err


module load python3/3.14.2
module load cuda/12.6.3

source venv/bin/activate

export DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/ufc10/}"

python3 -u src/train_early_fusion.py

