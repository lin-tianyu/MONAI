# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reproducible, lossless NIfTI/MLArray comparison."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import time
from pathlib import Path

import blosc2
import mlarray
import nibabel as nib
import numpy as np
import psutil
import torch

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import LoadImaged

TASKS = {"Task05_Prostate": (128, 128, 8), "Task09_Spleen": (128, 128, 16)}
FORMATS = ("nii.gz", "nii", "mla_lz4", "mla_zstd")


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def convert(src, dst, patch, codec):
    image = nib.load(src)
    array = image.dataobj.get_unscaled()
    axes = [{"name": name, "type": "space"} for name in ("x", "y", "z")]
    if array.ndim == 4:
        axes.append({"name": "c", "type": "channel", "labels": ["T2", "ADC"]})
    space = {
        "name": "patient",
        "axes": [
            {"name": n, "unit": "millimeter", "increases_toward": d}
            for n, d in zip(("x", "y", "z"), ("right", "anterior", "superior"))
        ],
        "from_array": {"array_axes": ["x", "y", "z"], "matrix": image.affine.tolist()},
    }
    slope, offset = image.dataobj.slope, image.dataobj.inter
    values = {"kind": "label" if "labelsTr" in str(src) else "measurement", "scale": slope, "offset": offset}
    params = None if codec == "mla_lz4" else {"codec": blosc2.Codec.ZSTD, "clevel": 5, "nthreads": 4}
    obj = mlarray.MLArray(
        np.ascontiguousarray(array),
        axes=axes,
        spaces=[space],
        values=values,
        patch_size=patch,
        cparams=params,
        source=[
            {
                "format": "nifti",
                "files": [str(src)],
                "reader": {"name": "nibabel", "version": nib.__version__},
                "header": {"binaryblock": image.header.binaryblock},
            }
        ],
    )
    obj.save(str(dst))
    obj.close()


def paths_for(root, task, kind, name):
    stem = name.removesuffix(".nii.gz")
    return {
        fmt: root
        / ("raw" if fmt == "nii.gz" else "preprocessed")
        / task
        / fmt
        / kind
        / (name if fmt == "nii.gz" else stem + (".nii" if fmt == "nii" else ".mla"))
        for fmt in FORMATS
    }


def prepare(args):
    root = args.data_root
    available = shutil.disk_usage(root.parent).free
    if available < 20 * 1024**3:
        raise RuntimeError("Need at least 20 GiB free before staging this experiment.")
    manifest, storage = [], []
    for task, patch in TASKS.items():
        files = sorted((args.source / task / "imagesTr").glob("*.nii.gz"))
        files = [p for p in files if not p.name.startswith(".")]
        files.sort(key=lambda p: (np.prod(nib.load(p).shape) * nib.load(p).get_data_dtype().itemsize, p.name))
        selected = [files[i] for i in np.linspace(0, len(files) - 1, min(args.cases, len(files)), dtype=int)]
        for src in selected:
            item = {"task": task, "case": src.name, "patch": patch, "shape": nib.load(src).shape[:3], "paths": {}}
            for kind in ("imagesTr", "labelsTr"):
                original = args.source / task / kind / src.name
                paths = paths_for(root, task, kind, src.name)
                for fmt, dst in paths.items():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    start = time.perf_counter()
                    reused = dst.exists()
                    if not reused:
                        temp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
                        if fmt == "nii.gz":
                            shutil.copyfile(original, temp)
                        elif fmt == "nii":
                            with gzip.open(original, "rb") as inp, temp.open("wb") as out:
                                shutil.copyfileobj(inp, out)
                        else:
                            convert(original, temp, patch, fmt)
                        temp.replace(dst)
                    storage.append(
                        dict(
                            task=task,
                            case=src.name,
                            kind=kind,
                            format=fmt,
                            bytes=dst.stat().st_size,
                            prepare_seconds=None if reused else time.perf_counter() - start,
                        )
                    )
                item["paths"][kind] = {fmt: str(p) for fmt, p in paths.items()}
            manifest.append(item)
            print("prepared", task, src.name, flush=True)
    save_json(args.output / "manifest.json", manifest)
    save_json(args.output / "storage.json", storage)
    return manifest


def read(path, slices=None):
    if path.endswith(".mla"):
        with mlarray.MLArray.open(path) as obj:
            array = obj.to_numpy() if slices is None else obj[slices]
            scale = 1 if obj.values.scale is None else obj.values.scale
            offset = 0 if obj.values.offset is None else obj.values.offset
            if scale != 1 or offset != 0:
                array = array.astype(np.float64) * scale + offset
    else:
        obj = nib.load(path, keep_file_open=True)
        array = np.asanyarray(obj.dataobj) if slices is None else obj.dataobj[slices]
    return np.array(array, order="C", copy=True)


def validate(manifest):
    loader = LoadImaged(["image", "label"], dtype=None, ensure_channel_first=True)
    for item in manifest:
        ref = {kind: read(paths["nii.gz"]) for kind, paths in item["paths"].items()}
        for fmt in FORMATS:
            data = loader({"image": item["paths"]["imagesTr"][fmt], "label": item["paths"]["labelsTr"][fmt]})
            for kind, key in (("imagesTr", "image"), ("labelsTr", "label")):
                arr = read(item["paths"][kind][fmt])
                np.testing.assert_array_equal(arr, ref[kind])
                rng = random.Random(2026)
                for _ in range(3):
                    sl = slices(item, rng)
                    np.testing.assert_array_equal(read(item["paths"][kind][fmt], sl), ref[kind][sl])
                assert arr.dtype == ref[kind].dtype
                np.testing.assert_allclose(
                    data[key].affine.numpy(), nib.load(item["paths"][kind]["nii.gz"]).affine, atol=1e-6
                )
        print("validated", item["task"], item["case"], flush=True)


def residency(path):
    result = subprocess.run(
        ["fincore", "--bytes", "--noheadings", "--output", "RES", path], capture_output=True, text=True
    )
    return int(result.stdout.strip()) if result.returncode == 0 else None


def evict(paths):
    verified = True
    for path in paths:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        verified &= residency(path) == 0
    return verified


def slices(item, rng):
    shape = item["shape"]
    size = tuple(min(a, b) for a, b in zip(shape, item["patch"]))
    starts = [rng.randrange(n - p + 1) for n, p in zip(shape, size)]
    return tuple(slice(s, s + p) for s, p in zip(starts, size))


def summary(times):
    return dict(median_ms=float(np.median(times) * 1000), p95_ms=float(np.percentile(times, 95) * 1000))


def micro(args, manifest):
    rng = random.Random(2026)
    rows = []
    for task in TASKS:
        cases = [i for i in manifest if i["task"] == task]
        trace = [(rng.choice(cases), None) for _ in range(args.patch_reads)]
        trace = [(item, slices(item, rng)) for item, _ in trace]
        for mode in ("full", "patch", "monai"):
            for cache in ("warm", "cold"):
                for fmt in FORMATS:
                    loader = LoadImaged(["image", "label"], dtype=None)
                    requests = [(i, None) for i in cases] if mode != "patch" else trace
                    times, verified = [], True
                    observations = []
                    for _repeat in range(args.repeats):
                        order = requests.copy()
                        rng.shuffle(order)
                        for item, sl in order:
                            paths = [item["paths"][k][fmt] for k in ("imagesTr", "labelsTr")]
                            if cache == "cold":
                                verified &= evict(paths)
                            else:
                                for p in paths:
                                    read(p)
                            start = time.perf_counter()
                            if mode == "monai":
                                out = loader(dict(zip(("image", "label"), paths)))
                                arrays = [np.array(out[k].numpy(), order="C", copy=True) for k in ("image", "label")]
                            else:
                                arrays = [read(p, sl) for p in paths]
                            times.append(time.perf_counter() - start)
                            observations.append(
                                dict(
                                    case=item["case"],
                                    repeat=_repeat,
                                    slices=None if sl is None else [[x.start, x.stop] for x in sl],
                                    seconds=times[-1],
                                )
                            )
                            del arrays
                            if mode == "monai":
                                del out
                    row = dict(
                        task=task,
                        format=fmt,
                        mode=mode,
                        cache=cache,
                        cold_verified=verified if cache == "cold" else None,
                        n=len(times),
                        total_seconds=sum(times),
                        **summary(times),
                    )
                    rows.append(row)
                    save_json(args.output / "micro.json", rows)
                    with (args.output / "raw_timings.jsonl").open("a") as f:
                        f.write(json.dumps(dict(group=row, observations=observations)) + "\n")
                    print("micro", row, flush=True)
    return rows


class PatchSource:
    def __init__(self, fmt, partial):
        self.fmt, self.partial = fmt, partial
        self.loader = LoadImaged(["image", "label"], dtype=None)

    def __call__(self, item):
        rng = random.Random(item["case"])
        coords = [slices(item, rng) for _ in range(4)]
        paths = [item["paths"][k][self.fmt] for k in ("imagesTr", "labelsTr")]
        out = []
        if not self.partial:
            full = self.loader(dict(zip(("image", "label"), paths)))
            arrays = [np.asarray(full[k]) for k in ("image", "label")]
            patches = [[a[sl] for a in arrays] for sl in coords]
        else:
            objects = [
                mlarray.MLArray.open(p) if p.endswith(".mla") else nib.load(p, keep_file_open=True) for p in paths
            ]
            try:
                patches = [
                    [
                        np.array(obj[sl] if p.endswith(".mla") else obj.dataobj[sl], copy=True)
                        for obj, p in zip(objects, paths)
                    ]
                    for sl in coords
                ]
            finally:
                for obj, p in zip(objects, paths):
                    if p.endswith(".mla"):
                        obj.close()
        for image, label in patches:
            if image.ndim == 3:
                image = image[None]
            else:
                image = np.moveaxis(image, -1, 0)
            out.append(
                {
                    "image": torch.from_numpy(np.array(image, dtype=np.float32, order="C", copy=True)),
                    "label": torch.from_numpy(np.array(label[None], dtype=np.int64, order="C", copy=True)),
                }
            )
        return out


def dataloaders(args, manifest):
    rows = []
    for task in TASKS:
        cases = [i for i in manifest if i["task"] == task]
        for fmt in FORMATS:
            for mode in ("full_crop", "partial", "ram_cache"):
                for workers in args.workers:
                    start = time.perf_counter()
                    transform = PatchSource(fmt, mode == "partial")
                    if mode == "ram_cache":
                        # Cache full volumes, then apply the same trace outside the cache.
                        load = LoadImaged(["image", "label"], dtype=None)
                        data = [
                            dict(image=i["paths"]["imagesTr"][fmt], label=i["paths"]["labelsTr"][fmt]) for i in cases
                        ]
                        cached = CacheDataset(data, transform=load, num_workers=1, progress=False)

                        class CachedPatches:
                            def __len__(self):
                                return len(cases)

                            def __getitem__(self, index):
                                full = cached[index]
                                rng = random.Random(cases[index]["case"])
                                out = []
                                for _ in range(4):
                                    sl = slices(cases[index], rng)
                                    im = np.asarray(full["image"])[sl]
                                    lab = np.asarray(full["label"])[sl]
                                    im = im[None] if im.ndim == 3 else np.moveaxis(im, -1, 0)
                                    out.append(
                                        dict(
                                            image=torch.from_numpy(
                                                np.array(im, dtype=np.float32, order="C", copy=True)
                                            ),
                                            label=torch.from_numpy(
                                                np.array(lab[None], dtype=np.int64, order="C", copy=True)
                                            ),
                                        )
                                    )
                                return out

                        dataset = CachedPatches()
                    else:
                        dataset = Dataset(cases, transform=transform)
                    init = time.perf_counter() - start
                    loader = DataLoader(dataset, batch_size=2, num_workers=workers, persistent_workers=workers > 0)
                    epochs = []
                    peak = 0
                    for _epoch in range(5):
                        start = time.perf_counter()
                        count = 0
                        for batch in loader:
                            count += len(batch["image"])
                            proc = psutil.Process()
                            peak = max(
                                peak,
                                sum(
                                    p.memory_info().rss
                                    for p in [proc] + proc.children(recursive=True)
                                    if p.is_running()
                                ),
                            )
                        epochs.append(time.perf_counter() - start)
                    rows.append(
                        dict(
                            task=task,
                            format=fmt,
                            mode=mode,
                            workers=workers,
                            init_seconds=init,
                            epoch_seconds=epochs,
                            patches_per_second=count / float(np.median(epochs[1:])),
                            summed_rss_bytes=peak,
                            children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                        )
                    )
                    save_json(args.output / "dataloader.json", rows)
                    print("loader", rows[-1], flush=True)
                    del loader, dataset
                    if mode == "ram_cache":
                        del cached
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the source datasets.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory for converted benchmark data.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["prepare", "validate", "micro", "loaders"])
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--patch-reads", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    save_json(
        args.output / "environment.json",
        dict(
            host=platform.node(),
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            nibabel=nib.__version__,
            mlarray=mlarray.__version__,
            blosc2=blosc2.__version__,
            read_ahead_supported=hasattr(blosc2, "set_prefetch"),
            argv=vars(args)
            | {"source": str(args.source), "data_root": str(args.data_root), "output": str(args.output)},
        ),
    )
    manifest = prepare(args) if "prepare" in args.phases else json.loads((args.output / "manifest.json").read_text())
    if "validate" in args.phases:
        validate(manifest)
        save_json(args.output / "validation.json", {"passed_pairs": len(manifest)})
    if "micro" in args.phases:
        micro(args, manifest)
    if "loaders" in args.phases:
        dataloaders(args, manifest)


if __name__ == "__main__":
    main()
