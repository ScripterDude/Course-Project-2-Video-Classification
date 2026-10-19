#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -J train_single_frame
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o train_single_frame_%J.out
#BSUB -e train_single_frame_%J.err
#BSUB -u s184208@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

module load python3/3.14.2
module load cuda/12.6.3

source venv/bin/activate

export DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/ufc10/}"

python3 -u src/train_single_frame.py

