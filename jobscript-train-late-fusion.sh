#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -J train_late_fusion
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 12:00
#BSUB -o train_late_fusion_%J.out
#BSUB -e train_late_fusion_%J.err

module load python3/3.9.25
module load cuda/12.6.3

source venv/bin/activate

export DATASET_VARIANT=original
export DATA_DIR="${DATA_DIR:-/dtu/datasets1/02516/ufc10/}"

echo "Python executable: $(which python)"
python --version

python -c "import numpy, torch, torchvision; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__)"

python -u src/train_late_fusion.py