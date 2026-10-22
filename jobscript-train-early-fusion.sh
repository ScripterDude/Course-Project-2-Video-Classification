#!/bin/bash
#BSUB -q c02516
#BSUB -app c02516_1g.10gb
#BSUB -J early_fusion
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
export DATASET_VARIANT="${DATASET_VARIANT:-noleakage}"

case "${TRAIN_MODE:-}" in
	search)
		wandb agent --count 1 "${WANDB_SWEEP_PATH:?Set WANDB_SWEEP_PATH to entity/project/sweep_id}"
		;;
	final)
		case "${FINAL_SEED:-}" in
			0|42|666) ;;
			*) echo "FINAL_SEED must be 0, 42, or 666."; exit 2 ;;
		esac
		python3 -u src/train_early_fusion.py
		;;
	test)
		python3 -u src/test_early_fusion.py
		;;
	*)
		echo "Set TRAIN_MODE to search, final, or test."
		exit 2
		;;
esac

