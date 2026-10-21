# Experiment protocol

## Datasets

- Use the provided train, validation, and test splits.
- Run the single-frame model separately on the original and no-leakage datasets.
- Run the dual-stream model on the no-leakage dataset.
- Compare models using the same dataset version and class mapping.
- Keep dataset versions in separate experiment directories.
- Select the dataset using DATASET_VARIANT: original or noleakage.
- Set DATA_DIR to the corresponding dataset directory.

## Models and preprocessing

### Single-frame ResNet50

- Initialize from ImageNet using ResNet50_Weights.DEFAULT.
- Use the preprocessing supplied by those weights.
- Train only layer4 and fc.
- Freeze parameters and batch-normalization statistics in earlier layers.
- Train on individual frames.
- During validation and testing, average class probabilities across the
  ten supplied frames to obtain one prediction per video.

### Dual-stream ResNet50

- Use ImageNet initialization for both streams.
- Initialize the flow input convolution by repeating the mean RGB weights.
- Train RGB layer4 and fc; train all flow layers.
- Subtract each flow channel's spatial mean.
- Clamp flow to [-28, 28] and divide by 28.
- Average RGB probabilities across frames.
- Select the RGB/flow fusion weight using validation data.

## Shared training settings

- Optimizer: Adam.
- Batch size: 32.
- Maximum epochs: 100.
- Early stopping: 10 epochs without a strict improvement in validation
  video accuracy.
- Scheduler: ReduceLROnPlateau, mode=max, factor=0.5, patience=3,
  minimum learning rate=1e-6.
- Default RGB learning rate: 3e-5.
- Default flow learning rate: 1e-4.
- Default weight decay: 1e-4.
- Set the training seed before model initialization.
- Record the actual pretrained weights, configuration, device, and runtime.

## Hyperparameter search

- Use Bayesian optimization with a maximum of 10 trials per model/dataset.
- Use training seed 0 for every search trial.
- Give each dataset its own sweep and selected configuration.
- Single-frame search:
  - Learning rate: log-uniform from 1e-5 to 1e-4.
  - Weight decay: log-uniform from 1e-6 to 1e-3.
- Dual-stream search:
  - RGB learning rate: log-uniform from 1e-5 to 1e-4.
  - Flow learning rate: log-uniform from 3e-5 to 3e-4.
  - Weight decay: log-uniform from 1e-6 to 1e-3.
- Keep other settings fixed.
- Select the configuration with the highest validation video accuracy.
- Single-frame selection uses RGB accuracy.
- Dual-stream selection uses fused accuracy.
- Keep the earliest checkpoint when validation accuracy ties.
- Use only training and validation data during search.

## Final training and testing

1. Finish the search for the selected model/dataset.
2. Select its best validation configuration.
3. Train that configuration from pretrained initialization with
   seeds 0, 42, and 666.
4. Select each seed's checkpoint using validation accuracy.
5. Evaluate all three checkpoints on the fixed test split.

These are repeated training runs on fixed splits, not cross-validation.

The single-frame budget across both datasets is 20 search runs and
6 final runs. Record actual elapsed time as well as run counts.

## Reporting

For each model/dataset, save:

- Configuration, epoch history, training plots, and best checkpoint.
- Predictions for every test video and training seed.
- Overall test accuracy for each seed.
- Per-class and per-(action, group) accuracy.
- Confusion matrices for individual seeds.
- Overall, per-class, and per-group mean and sample standard deviation
  across seeds.
- Mean and sample standard deviation of row-normalized confusion matrices.
- A heatmap of the mean normalized confusion matrix.

Use sample standard deviation with ddof=1.
Keep accuracy values as fractions in CSV files.
Display percentages in report tables and plots.

Report the architecture, pretrained weights, trainable layers, selected
hyperparameters, dataset version, search budget, and mean ± standard
deviation over seeds. State that hyperparameters were tuned separately
for the two datasets.

Training metrics are logged to W&B and local files.
Test reports are saved locally.

## Experiment organization

Single-frame results:

experiments/SingleFrameResNet50/<dataset>/<search|final>/<run_id>_seed<seed>/

Existing dual-stream results:

experiments/DualStreamResNet50/<search|final>/<run_id>_seed<seed>/

Use one final run per seed in each model/dataset's final directory.
Keep unrelated searches and earlier final runs outside the directories
used for selection and reporting.