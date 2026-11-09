# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Benchmark MONAI's normal whole-image load and random crop pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
)
from monai.utils import set_determinism


def make_transform(task, patch, augment=False):
    transforms = [
        LoadImaged(keys=("image", "label"), dtype=None),
        EnsureChannelFirstd(keys="image", channel_dim=-1 if task == "Task05_Prostate" else "no_channel"),
        EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
    ]
    if augment and task == "Task05_Prostate":
        transforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    transforms.extend(
        [
            RandSpatialCropSamplesd(keys=("image", "label"), roi_size=patch, num_samples=4, random_size=False),
            EnsureTyped(keys=("image", "label"), dtype=(torch.float32, torch.int64)),
        ]
    )
    if augment:
        transforms.extend(
            [
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
                RandRotate90d(keys=("image", "label"), prob=0.5, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandGaussianNoised(keys="image", prob=0.15, std=0.01),
            ]
        )
    return Compose(transforms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--formats", nargs="+", default=["nii.gz", "nii", "mla_lz4", "mla_zstd"])
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cases", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=["Task05_Prostate", "Task09_Spleen"])
    parser.add_argument("--augment", action="store_true", help="Apply MONAI spatial and intensity augmentation.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    torch.set_num_threads(1)
    for task in args.tasks:
        cases = [item for item in manifest if item["task"] == task][: args.cases]
        if not cases:
            raise ValueError(f"No cases found for {task}")
        for fmt in args.formats:
            for cached in (False, True):
                for workers in args.workers:
                    set_determinism(seed=2026)
                    data = [
                        {"image": item["paths"]["imagesTr"][fmt], "label": item["paths"]["labelsTr"][fmt]}
                        for item in cases
                    ]
                    transform = make_transform(task, cases[0]["patch"], args.augment)
                    start = time.perf_counter()
                    dataset = (
                        CacheDataset(data, transform=transform, num_workers=1, progress=False)
                        if cached
                        else Dataset(data, transform=transform)
                    )
                    init = time.perf_counter() - start
                    loader = DataLoader(dataset, batch_size=2, num_workers=workers, persistent_workers=workers > 0)
                    epochs = []
                    for _ in range(args.epochs):
                        start = time.perf_counter()
                        count = 0
                        for batch in loader:
                            count += len(batch["image"])
                        epochs.append(time.perf_counter() - start)
                    steady = epochs[1:] or epochs
                    row = dict(
                        task=task,
                        format=fmt,
                        cached=cached,
                        workers=workers,
                        cases=len(cases),
                        augmented=args.augment,
                        init_seconds=init,
                        epoch_seconds=epochs,
                        patches_per_second=count / float(np.median(steady)),
                    )
                    rows.append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(rows, indent=2))
                    print(json.dumps(row), flush=True)
                    del loader, dataset


if __name__ == "__main__":
    main()
