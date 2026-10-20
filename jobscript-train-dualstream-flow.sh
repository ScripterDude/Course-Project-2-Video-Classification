#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_2g.20gb
#BSUB -J train_dualstream_flow
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o train_dualstream_flow_%J.out
#BSUB -e train_dualstream_flow_%J.err


module load python3/3.14.2
module load cuda/12.6.3

source venv/bin/activate

export DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/ucf101noleakage}"

python3 -u src/train_dualstream_flow.py

