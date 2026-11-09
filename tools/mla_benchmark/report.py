# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Export raw tables, figures and a Chinese report from a completed run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import mlarray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output
    tables = {name: json.loads((root / f"{name}.json").read_text()) for name in ("storage", "micro", "dataloader")}
    for name, rows in tables.items():
        with (root / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    layouts = []
    for item in json.loads((root / "manifest.json").read_text()):
        for kind, paths in item["paths"].items():
            for fmt in ("mla_lz4", "mla_zstd"):
                with mlarray.MLArray.open(paths[fmt]) as obj:
                    layouts.append(
                        dict(
                            task=item["task"],
                            case=item["case"],
                            kind=kind,
                            format=fmt,
                            storage=obj.meta.to_dict()["storage"],
                            shape=obj.shape,
                            dtype=str(obj.dtype),
                        )
                    )
    (root / "layouts.json").write_text(json.dumps(layouts, indent=2))
    sizes = defaultdict(int)
    conversion = defaultdict(float)
    unknown_conversion = set()
    for r in tables["storage"]:
        sizes[r["task"], r["format"]] += r["bytes"]
        if r["format"].startswith("mla"):
            if r["prepare_seconds"] is None:
                unknown_conversion.add((r["task"], r["format"]))
            else:
                conversion[r["task"], r["format"]] += r["prepare_seconds"]
    tasks = sorted({r["task"] for r in tables["storage"]})
    formats = ["nii.gz", "nii", "mla_lz4", "mla_zstd"]
    task_cases = {task: len({r["case"] for r in tables["storage"] if r["task"] == task}) for task in tasks}
    if task_cases.get("Task09_Spleen", 0) >= 41:
        spleen_gib = sizes["Task09_Spleen", "nii"] / 2**30
        spleen_note = (
            f"本次包含 Spleen 完整 41 对训练病例；实验生成的未压缩 `.nii` 图像和标签合计 {spleen_gib:.2f} GiB。"
            "这在本机仍可放入内存，不能代表超过内存容量的训练集。"
        )
    else:
        spleen_note = (
            f"本次 Spleen 只测了 {task_cases.get('Task09_Spleen', 0)} 对；完整训练集有 41 对。"
            "因此这仍是抽样初筛，不能代表完整数据集，更不能外推到几十或几百 GiB 的训练集。"
        )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    text = [
        "# MLA 与 NIfTI 读取实验：先看结论",
        "",
        "## 这次做了什么",
        "",
        f"对 {len({(r['task'], r['case']) for r in tables['storage']})} 个真实 image/label 病例对，逐一比较现有 `.nii.gz`、解压后的 `.nii`、LZ4 MLA 和 ZSTD-5 MLA。",
        "转换保留体素值、数据类型和空间方向。CSV/JSON 保存汇总和逐次计时；两张 PNG 是体积、读取延迟和 DataLoader 吞吐图。",
        "",
        "## 结论怎么读",
        "",
        "文件更小表示省训练副本空间；读取毫秒更低或每秒 patch 数更高表示 CPU 供数更快。patch 指训练时从一个三维体积裁出的较小子区域。",
        "",
        "## 测试时的缓存是什么意思",
        "",
        "操作系统会把最近读过的文件内容暂存在内存里。‘热缓存’表示计时前先读一遍文件，让数据大概率已经在内存中；‘冷缓存’表示测试前要求系统丢弃该文件的内存副本，并检查文件页是否还驻留。前者模拟重复读取，后者模拟重新从 SSD 读取。它们不是文件格式或 GPU 缓存；未通过检查的冷缓存组不作为冷读取结论。",
        "",
        "## 本轮病例数",
        "",
    ]
    source_case_counts = {"Task05_Prostate": 32, "Task09_Spleen": 41}
    for task in tasks:
        text.append(
            f"- {task}：测试 {task_cases[task]}/{source_case_counts.get(task, task_cases[task])} 对 image/label。"
        )
    text += ["", "## 结论先看", ""]
    for task in tasks:
        zstd_ratio = sizes[task, "mla_zstd"] / sizes[task, "nii.gz"]
        throughput = {}
        for fmt in ("nii.gz", "nii", "mla_lz4", "mla_zstd"):
            throughput[fmt] = {}
            for mode in ("full_crop", "partial"):
                throughput[fmt][mode] = max(
                    r["patches_per_second"]
                    for r in tables["dataloader"]
                    if r["task"] == task and r["format"] == fmt and r["mode"] == mode
                )
        if zstd_ratio < 1:
            text.append(
                f"- {task}：ZSTD MLA 占 `.nii.gz` 的 {zstd_ratio:.2f} 倍。标准 MONAI 整幅读取再裁块的吞吐，"
                f"ZSTD 为 {throughput['mla_zstd']['full_crop']:.1f}、`.nii.gz` 为 {throughput['nii.gz']['full_crop']:.1f} patches/s。"
                f"实验中的按需 patch 路线，ZSTD 为 {throughput['mla_zstd']['partial']:.1f}、LZ4 为 "
                f"{throughput['mla_lz4']['partial']:.1f}、未压缩 `.nii` 为 {throughput['nii']['partial']:.1f}、"
                f"`.nii.gz` 为 {throughput['nii.gz']['partial']:.1f} patches/s。"
            )
        else:
            text.append(f"- {task}：ZSTD MLA 占 `.nii.gz` 的 {zstd_ratio:.2f} 倍，存储没有收益。")
    text += [
        "",
        "关键限制：当前 `MLArrayReader` 配合 `LoadImaged` 会读入整个体积；上面的“整幅读取再裁块”是可直接使用的 MONAI 路径。“按需 patch”使用 benchmark 专用 loader，直接用 MLArray 切片，还没有接入通用 `LoadImaged` transform，不能把这条吞吐直接当作现有 MONAI pipeline 的结果。",
        "这次证明完整 `.mla` 读取可在 MONAI 中使用；如需训练时只从磁盘取小块，还需后续增加 patch-aware transform/loader。所有吞吐都是 CPU 供数指标，不是 GPU 训练速度。",
        "",
    ]
    text += [
        "",
        "病例数不等于数据量大小。" + spleen_note,
        "",
        "# 详细数据",
        "",
        "进程内存统计可能重复计算共享内存页，因此只用于比较配置。吞吐是 CPU 供数指标，不是 GPU 训练加速。",
        "",
        "| 数据集 | 格式 | image+label MiB | 相对 nii.gz 大小 | 热缓存整幅读取 ms | 热缓存 patch ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, ax in zip(tasks, axes):
        for fmt in formats:
            rows = [r for r in tables["micro"] if r["task"] == task and r["format"] == fmt and r["cache"] == "warm"]
            full = next(r["median_ms"] for r in rows if r["mode"] == "full")
            patch = next(r["median_ms"] for r in rows if r["mode"] == "patch")
            ax.scatter(sizes[task, fmt] / 2**20, patch, label=fmt)
            text.append(
                f'| {task} | {fmt} | {sizes[task,fmt]/2**20:.2f} | {sizes[task,fmt]/sizes[task,"nii.gz"]:.3f} | {full:.3f} | {patch:.3f} |'
            )
        ax.set(title=task, xlabel="Image + label size (MiB)", ylabel="Warm patch median (ms)")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "size_latency.png", dpi=180)
    plt.close(fig)
    text += ["", "## 相对压缩 NIfTI 的双收益判断", ""]
    for task in tasks:

        def latency(fmt, mode):
            return next(
                r["median_ms"]
                for r in tables["micro"]
                if r["task"] == task and r["format"] == fmt and r["mode"] == mode and r["cache"] == "warm"
            )

        ratio = sizes[task, "mla_zstd"] / sizes[task, "nii.gz"]
        full_speed = latency("nii.gz", "full") / latency("mla_zstd", "full")
        patch_speed = latency("nii.gz", "patch") / latency("mla_zstd", "patch")
        verdict = (
            "本轮在体积与读取延迟上均有改善"
            if ratio < 1 and full_speed > 1 and patch_speed > 1
            else "本轮未同时满足更快和更小"
        )
        text.append(
            f"- {task}：ZSTD-5 MLA 大小为 nii.gz 的 {ratio:.3f} 倍，热缓存整幅读取约 {full_speed:.2f} 倍速、随机 patch 约 {patch_speed:.2f} 倍速；{verdict}。"
        )
    text += [
        "",
        "patch 微测试每次重新打开文件；DataLoader 则每个病例打开一次再取四个 patch，可观察重复读取时的开销。DataLoader 包括整幅读取后取 patch、只读 patch 区域和先缓存整幅数据三种方式。所有这些数据集均可放入本机 RAM，因此不代表超内存工作负载。",
        "",
    ]
    text += ["", "| 数据集 | 格式 | 路线 | 最佳 workers | patches/s |", "|---|---|---|---:|---:|"]
    for task in tasks:
        for fmt in formats:
            for mode in ("full_crop", "partial", "ram_cache"):
                row = max(
                    (r for r in tables["dataloader"] if r["task"] == task and r["format"] == fmt and r["mode"] == mode),
                    key=lambda r: r["patches_per_second"],
                )
                text.append(f'| {task} | {fmt} | {mode} | {row["workers"]} | {row["patches_per_second"]:.1f} |')
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row_index, task in enumerate(tasks):
        for col_index, mode in enumerate(("full_crop", "partial")):
            ax = axes[row_index, col_index]
            for fmt in formats:
                records = sorted(
                    (r for r in tables["dataloader"] if r["task"] == task and r["mode"] == mode and r["format"] == fmt),
                    key=lambda r: r["workers"],
                )
                ax.plot(
                    [r["workers"] for r in records], [r["patches_per_second"] for r in records], marker="o", label=fmt
                )
            ax.set(title=f"{task}: {mode}", xlabel="DataLoader workers", ylabel="Patches/s")
            ax.legend()
    fig.tight_layout()
    fig.savefig(root / "dataloader_throughput.png", dpi=180)
    plt.close(fig)
    cold = [r for r in tables["micro"] if r["cache"] == "cold"]
    cold_failed = [r for r in cold if not r["cold_verified"]]
    text += [
        "",
        f'冷缓存验证：{sum(bool(r["cold_verified"]) for r in cold)}/{len(cold)} 个实验组的每次读取均确认文件页已清出内存。',
        "未通过的组："
        + (", ".join(f'{r["task"]}/{r["format"]}/{r["mode"]}' for r in cold_failed) if cold_failed else "无。"),
        "",
        "转换回本按“转换总时间 / 同一遍数据读取节省时间”计算；逐次原始延迟保存在 raw_timings.jsonl，可复核波动。",
        "",
    ]
    for task in tasks:
        for fmt in ("mla_lz4", "mla_zstd"):

            def total(f):
                return next(
                    r["total_seconds"]
                    for r in tables["micro"]
                    if r["task"] == task and r["format"] == f and r["mode"] == "full" and r["cache"] == "warm"
                )

            base = next(
                r
                for r in tables["micro"]
                if r["task"] == task and r["format"] == "nii.gz" and r["mode"] == "full" and r["cache"] == "warm"
            )
            repeats = base["n"] / len(
                [
                    r
                    for r in tables["storage"]
                    if r["task"] == task and r["format"] == "nii.gz" and r["kind"] == "imagesTr"
                ]
            )
            if (task, fmt) in unknown_conversion:
                text.append(f"- {task} / {fmt}：复用了已转换文件，本次无法估算转换回本次数。")
                continue
            saved = (total("nii.gz") - total(fmt)) / repeats
            text.append(
                f"- {task} / {fmt}："
                + (
                    f"约 {conversion[task,fmt]/saved:.1f} 遍整幅读取回本。"
                    if saved > 0
                    else "整幅读取没有节省时间，无法回本。"
                )
            )
    (root / "REPORT.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
