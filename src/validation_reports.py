from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def _save_cm(cm, path, title, percent=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        100 * cm if percent else cm,
        cmap="Blues", vmin=0, vmax=100 if percent else None,
    )
    ax.set(
        title=title, xlabel="Predicted class", ylabel="True class",
        xticks=range(len(cm)), yticks=range(len(cm)),
    )
    fig.colorbar(image, ax=ax, label="Videos (%)" if percent else "Videos")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_validation_reports(model, rgb_loader, run_dir, device, flow_loader=None, clip_model=False):
    run_dir = Path(run_dir)
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True,
    )
    config = checkpoint["model_config"]
    model.load_state_dict(checkpoint["model_state_dict"])

    if flow_loader is not None:
        model.rgb_weight = checkpoint["rgb_weight"]

    del checkpoint
    model.eval()

    dataset = rgb_loader.dataset
    if not len(dataset):
        raise ValueError("The validation split is empty.")

    if flow_loader is not None:
        if dataset.video_paths != flow_loader.dataset.video_paths:
            raise ValueError("RGB and flow validation videos do not match.")

    video_names = [Path(path).stem for path in dataset.video_paths]
    metadata = dataset.df.set_index("video_name", verify_integrity=True)
    video_info = metadata.loc[video_names, ["action"]].reset_index()
    video_info["group"] = (
        metadata.loc[video_names, "group"].to_numpy()
        if "group" in metadata.columns
        else video_info["video_name"].str.extract(r"_(g\d+)_", expand=False)
    )

    streams = ["rgb"] if flow_loader is None else ["rgb", "flow", "fused"]
    batches = (
        ((batch, None) for batch in rgb_loader)
        if flow_loader is None else zip(rgb_loader, flow_loader)
    )

    rows = []
    with torch.inference_mode():
        for (videos, labels), flow_batch in batches:
            videos = videos.to(device)
            if clip_model:   # fusion models take the whole clip [B, 3, T, H, W]
                probabilities = {"rgb": model(videos).softmax(dim=1)}
            else:
                b, c, t, h, w = videos.shape
                frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                rgb_model = model if flow_loader is None else model.rgb_stream
                probabilities = {
                    "rgb": (
                        rgb_model(frames).softmax(dim=1)
                        .reshape(b, t, -1).mean(dim=1)
                    )
                }

            if flow_batch is not None:
                flows, flow_labels = flow_batch
                if not torch.equal(labels, flow_labels):
                    raise ValueError("RGB and flow validation labels do not match.")

                flows = flows.to(device)
                if config.get("subtract_flow_mean", False):
                    flows = flows - flows.mean(dim=(-2, -1), keepdim=True)

                bound = config["flow_bound"]
                flows = flows.clamp(-bound, bound) / bound
                probabilities["flow"] = model.flow_stream(flows).softmax(dim=1)
                probabilities["fused"] = (
                    model.rgb_weight * probabilities["rgb"]
                    + (1 - model.rgb_weight) * probabilities["flow"]
                )

            rows.extend(zip(
                labels.tolist(),
                *[probabilities[s].argmax(dim=1).cpu().tolist() for s in streams],
            ))

    if len(rows) != len(video_info):
        raise ValueError("Validation predictions do not match the video count.")

    predictions = pd.concat([
        video_info,
        pd.DataFrame(
            rows, columns=["true_label"] + [f"{s}_pred" for s in streams],
        ),
    ], axis=1)
    predictions["seed"] = config["seed"]

    output = run_dir / "val"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)

    summary = {"seed": config["seed"]}
    for stream in streams:
        result = predictions.assign(
            correct=predictions[f"{stream}_pred"].eq(predictions["true_label"])
        )
        summary[stream] = result["correct"].mean()

        for report, columns in [
            ("per_class", ["true_label", "action"]),
            ("per_group", ["action", "group"]),
        ]:
            result.groupby(columns)["correct"].agg(
                total="size", accuracy="mean",
            ).to_csv(output / f"{stream}_{report}.csv")

        cm = pd.crosstab(
            result["true_label"], result[f"{stream}_pred"],
        ).reindex(
            index=range(config["num_classes"]),
            columns=range(config["num_classes"]), fill_value=0,
        )
        cm.to_csv(output / f"{stream}_confusion_matrix.csv")
        _save_cm(
            cm, output / f"{stream}_confusion_matrix.png",
            f"{stream.upper()} validation - seed {config['seed']}",
        )

    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    return output


def summarize_validation_reports(report_dirs):
    report_dirs = [Path(path) for path in report_dirs]
    output = report_dirs[0].parent.parent / "val"
    output.mkdir(parents=True, exist_ok=True)

    per_seed = pd.concat([
        pd.read_csv(path / "summary.csv") for path in report_dirs
    ]).set_index("seed")
    per_seed.to_csv(output / "val_per_seed.csv")

    # Pandas std uses sample standard deviation: ddof=1.
    stats = per_seed.agg(["mean", "std"]).T
    stats.to_csv(output / "val_mean_std.csv")

    for stream, row in stats.iterrows():
        print(
            f"Validation {stream.upper()}: "
            f"{100 * row['mean']:.2f}% +/- {100 * row['std']:.2f}%"
        )

        for report, columns in [
            ("per_class", ["true_label", "action"]),
            ("per_group", ["action", "group"]),
        ]:
            tables = pd.concat([
                pd.read_csv(path / f"{stream}_{report}.csv")
                for path in report_dirs
            ])
            tables.groupby(columns).agg(
                total=("total", "first"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
            ).to_csv(output / f"{stream}_{report}_mean_std.csv")

        matrices = [
            pd.read_csv(path / f"{stream}_confusion_matrix.csv", index_col=0)
            for path in report_dirs
        ]
        normalized = pd.concat([
            cm.div(cm.sum(axis=1).replace(0, float("nan")), axis=0)
            for cm in matrices
        ])
        mean = normalized.groupby(level=0).mean()
        std = normalized.groupby(level=0).std(ddof=1)

        mean.to_csv(output / f"{stream}_confusion_matrix_mean.csv")
        std.to_csv(output / f"{stream}_confusion_matrix_std.csv")
        _save_cm(
            mean, output / f"{stream}_confusion_matrix_mean.png",
            f"{stream.upper()} validation - mean over {len(report_dirs)} seeds",
            percent=True,
        )