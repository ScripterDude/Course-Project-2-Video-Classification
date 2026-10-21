from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import json
from torchvision.models import ResNet50_Weights
from datasets import FlowVideoDataset, FrameVideoDataset, root_dir
from models import DualStreamResNet50

#find the best val video acc
model_name = "DualStreamResNet50"
model_dir = Path("experiments") / model_name/"final"
root_dir = os.getenv("DATA_DIR", "Data/ucf101_noleakage")
seeds = [0, 42, 666]

runs = {}
for config_path in model_dir.glob("*/config.json"):
    with open(config_path, encoding="utf-8") as f:
        run_config = json.load(f)
    runs[run_config["seed"]] = (config_path.parent, run_config)

seed_results = []

for seed in seeds:
    experiment_dir, config = runs[seed]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(experiment_dir / "best_model.pt",map_location=device,weights_only=True)
    model = DualStreamResNet50(num_classes=config["num_classes"], flow_channels=config["flow_channels"], rgb_weight=config["rgb_weight"], weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # load up test
    rgb_transform = ResNet50_Weights[config["weights"]].transforms()

    rgb_test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=rgb_transform, stack_frames=True)
    flow_test_dataset = FlowVideoDataset(root_dir=root_dir, split="test")

    #for more informative stats
    metadata = rgb_test_dataset.df.set_index("video_name", verify_integrity=True)
    video_names = [Path(path).stem for path in rgb_test_dataset.video_paths]
    video_info = metadata.loc[video_names, ["action", "group", "label"]].reset_index()

    rgb_test_loader = DataLoader(rgb_test_dataset, batch_size=config["batch_size"], shuffle=False)
    flow_test_loader = DataLoader(flow_test_dataset, batch_size=config["batch_size"], shuffle=False)

    prediction_rows = []
    rgb_correct = 0
    flow_correct = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for (videos, labels), (flows, flow_labels) in zip(rgb_test_loader, flow_test_loader):
            if not torch.equal(labels, flow_labels):
                raise ValueError("RGB and flow test labels do not match.")

            videos = videos.to(device)
            if config.get("subtract_flow_mean", False):
                flows = flows - flows.mean(dim=(-2, -1), keepdim=True)
            flows = flows.to(device).clamp(-config["flow_bound"], config["flow_bound"]) / config["flow_bound"]
            labels = labels.to(device)

            # [B, 3, T, H, W] -> [B*T, 3, H, W]
            b, c, t, h, w = videos.shape
            frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

            # mean RGB probabilities across frames then fuse with flow
            rgb_probs = model.rgb_stream(frames).softmax(dim=1).reshape(b, t, -1).mean(dim=1)
            flow_probs = model.flow_stream(flows).softmax(dim=1)
            probabilities = model.rgb_weight * rgb_probs + (1.0 - model.rgb_weight) * flow_probs

            rgb_correct += (rgb_probs.argmax(dim=1) == labels).sum().item()
            flow_correct += (flow_probs.argmax(dim=1) == labels).sum().item()
            correct += (probabilities.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            prediction_rows.extend(zip(labels.cpu().tolist(),rgb_probs.argmax(dim=1).cpu().tolist(),flow_probs.argmax(dim=1).cpu().tolist(), probabilities.argmax(dim=1).cpu().tolist()))

    # save the detailed results for this seed
    predictions = pd.DataFrame(prediction_rows,columns=["true_label", "rgb_pred", "flow_pred", "fused_pred"])
    predictions = pd.concat([video_info[["video_name", "action", "group"]], predictions],axis=1)
    predictions["seed"] = seed
    predictions.to_csv(experiment_dir / "predictions.csv", index=False)

    seed_result = {"seed": seed}
    num_classes = config["num_classes"]

    for stream in ["rgb", "flow", "fused"]:
        result = predictions.assign(correct=predictions[f"{stream}_pred"].eq(predictions["true_label"]))
        seed_result[stream] = result["correct"].mean()

        result.groupby(["true_label", "action"])["correct"].agg(total="size", accuracy="mean",).to_csv(experiment_dir / f"{stream}_per_class.csv")

        result.groupby(["action", "group"])["correct"].agg(total="size", accuracy="mean",).to_csv(experiment_dir / f"{stream}_per_group.csv")

        #confusion matrix
        cm = pd.crosstab(result["true_label"], result[f"{stream}_pred"],).reindex(index=range(num_classes),columns=range(num_classes),fill_value=0)
        cm.to_csv(experiment_dir / f"{stream}_confusion_matrix.csv")

        fig, ax = plt.subplots(figsize=(7, 6))
        heatmap = ax.imshow(cm, cmap="Blues")
        ax.set(
            title=f"{stream.upper()} - seed {seed}",
            xlabel="Predicted class",
            ylabel="True class",
            xticks=range(num_classes),
            yticks=range(num_classes),
        )
        fig.colorbar(heatmap, ax=ax)
        fig.savefig(experiment_dir / f"{stream}_confusion_matrix.png",bbox_inches="tight")
        plt.close(fig)

    seed_results.append(seed_result)
    pd.DataFrame([seed_result]).to_csv(experiment_dir / "summary.csv", index=False)

per_seed = pd.DataFrame(seed_results).set_index("seed")
per_seed.to_csv(model_dir / "test_per_seed.csv")

stats = per_seed.agg(["mean", "std"]).T
stats.to_csv(model_dir / "test_mean_std.csv")

for stream, row in stats.iterrows():
    print(f"{stream.upper()}: {100 * row['mean']:.2f}% +- {100 * row['std']:.2f}%")

for stream in ["rgb", "flow", "fused"]:
    # class and group acroess seds
    for report, group_cols in [("per_class", ["true_label", "action"]),("per_group", ["action", "group"]),]:
        tables = [pd.read_csv(runs[seed][0] / f"{stream}_{report}.csv")for seed in seeds]

        summary = pd.concat(tables).groupby(group_cols).agg(total=("total", "first"),accuracy_mean=("accuracy", "mean"),accuracy_std=("accuracy", "std"))
        summary.to_csv(model_dir / f"{stream}_{report}_mean_std.csv")

    # normalize each seed confusion matrix by its true class counts
    matrices = [pd.read_csv(runs[seed][0] / f"{stream}_confusion_matrix.csv",index_col=0,) for seed in seeds]
    matrices = pd.concat([cm.div(cm.sum(axis=1), axis=0)for cm in matrices])

    cm_mean = matrices.groupby(level=0).mean()
    cm_std = matrices.groupby(level=0).std()

    cm_mean.to_csv(model_dir / f"{stream}_confusion_matrix_mean.csv")
    cm_std.to_csv(model_dir / f"{stream}_confusion_matrix_std.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.imshow(100 * cm_mean, cmap="Blues", vmin=0, vmax=100)
    ax.set(title=f"{stream.upper()} - mean over {len(seeds)} seeds",xlabel="Predicted class",ylabel="True class",xticks=range(num_classes),yticks=range(num_classes))
    fig.colorbar(heatmap, ax=ax, label="Videos (%)")
    fig.savefig(model_dir / f"{stream}_confusion_matrix_mean.png",bbox_inches="tight")
    plt.close(fig)