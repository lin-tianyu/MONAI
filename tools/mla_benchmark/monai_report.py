# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Summarize the MONAI-native pipeline benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("storage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    sizes = defaultdict(int)
    for row in json.loads(args.storage.read_text()):
        sizes[row["task"], row["format"]] += row["bytes"]
    tasks = sorted({row["task"] for row in rows})
    augmented = bool(rows[0].get("augmented", False))
    lines = [
        "# MONAI 原生训练数据管线对照",
        "",
        "本实验使用 MONAI `LoadImaged` 读取整幅图像，`EnsureChannelFirstd` 调整通道，"
        "`RandSpatialCropSamplesd` 每个病例随机裁四个图像和标签小块，再由 MONAI `DataLoader` 供数。"
        "另用 `CacheDataset` 把确定性的加载结果预先放入内存。",
        (
            "Prostate MRI 先对每个通道的非零体素进行强度标准化。数据增强使用 MONAI 随机翻转和 90° 平面旋转，图像与标签同步变换；"
            "随机强度缩放、偏移和高斯噪声只作用于图像。这些参数用于格式对照，不代表特定临床任务的最佳增强方案。"
            if augmented
            else "本轮仅随机裁块，未启用额外的数据增强。"
        ),
        "",
        "worker 是并行加载数据的进程；一个 epoch 表示遍历一遍病例。每种格式、每种缓存方式和 worker 数均使用相同随机种子和病例顺序；每组运行 5 个 epoch，"
        "下面的速度取后 4 个 epoch 的中位数。缓存准备时间单独记录；未使用 CacheDataset 的测试也可能受操作系统文件缓存影响。"
        "文件大小包含图像和标签；Spleen 的 41 对病例均可放进本机内存。测试没有运行 GPU 训练，也没有控制操作系统缓存是否已有文件。",
        "",
        "## 主要发现",
        "",
    ]
    for task in tasks:

        def get(fmt, cached):
            return next(
                r
                for r in rows
                if r["task"] == task and r["format"] == fmt and r["cached"] == cached and r["workers"] == 4
            )

        base = get("nii.gz", False)
        zstd = get("mla_zstd", False)
        ratio = sizes[task, "mla_zstd"] / sizes[task, "nii.gz"]
        lines.append(
            f"- {task}：4 workers、每次加载完整图像的速度：`.nii.gz` {base['patches_per_second']:.1f}，"
            f"ZSTD MLA {zstd['patches_per_second']:.1f} 个小块/秒；ZSTD 文件总大小是 `.nii.gz` 的 {ratio:.2f} 倍。"
        )
    lines += [
        "",
        "该 MONAI 标准管线会完整读取体积后再裁块。先前实验的按需 MLA 切片使用专用 loader，"
        "它的数字不等于这条标准管线的结果；若要让 MONAI 直接从文件读取小块，还需进一步开发。",
        "",
        "## 各配置结果",
        "",
        "缓存：否表示每个 epoch 重新通过 MONAI 加载；是表示先把整幅图像载入内存。"
        "缓存初始化时间不计入下表吞吐。完整逐 epoch 时间见 results.json。",
        "",
        "| 数据集 | 格式 | 放入内存 | workers | 病例对 | 缓存准备秒数 | 小块/秒 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['format']} | {'是' if r['cached'] else '否'} | {r['workers']} | {r['cases']} | "
            f"{r['init_seconds']:.2f} | {r['patches_per_second']:.1f} |"
        )
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
