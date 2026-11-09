# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from monai.data import DataLoader, Dataset, MLArrayReader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
)
from monai.utils import optional_import

mla, has_mla = optional_import("mlarray", version="1.0.0")


def image(array, affine, directions=("right", "anterior", "superior"), order=("x", "y", "z"), scale=1, offset=0):
    axes = [{"name": n, "type": "space"} for n in ("x", "y", "z")]
    if array.ndim == 4:
        axes.append({"name": "c", "type": "channel"})
    space = {
        "name": "patient",
        "axes": [{"name": n, "unit": "millimeter", "increases_toward": d} for n, d in zip(("x", "y", "z"), directions)],
        "from_array": {"array_axes": list(order), "matrix": affine.tolist()},
    }
    return mla.MLArray(array, axes=axes, spaces=[space], values={"scale": scale, "offset": offset})


@unittest.skipUnless(has_mla, "MLArray v1 required")
class TestMLArrayReader(unittest.TestCase):
    def test_mri_augmentation_keeps_image_and_label_aligned(self):
        labels = np.indices((8, 8, 8)).sum(axis=0).astype(np.uint8) % 2
        mri = np.stack([labels, labels], axis=-1).astype(np.float32)
        affine = np.diag([1.5, 2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            image_path, label_path = Path(tmp) / "image.mla", Path(tmp) / "label.mla"
            image(mri, affine).save(image_path)
            image(labels, affine).save(label_path)
            pipeline = Compose(
                [
                    LoadImaged(["image", "label"], dtype=None),
                    EnsureChannelFirstd(["image", "label"]),
                    RandSpatialCropSamplesd(["image", "label"], roi_size=(4, 4, 4), num_samples=2, random_size=False),
                    RandFlipd(["image", "label"], prob=1, spatial_axis=0),
                    RandRotate90d(["image", "label"], prob=1, spatial_axes=(0, 1)),
                    RandScaleIntensityd("image", factors=0.1, prob=1),
                    RandShiftIntensityd("image", offsets=0.1, prob=1),
                    RandGaussianNoised("image", std=0.01, prob=1),
                ]
            ).set_random_state(seed=2026)
            for sample in pipeline({"image": str(image_path), "label": str(label_path)}):
                self.assertEqual(tuple(sample["image"].shape), (2, 4, 4, 4))
                np.testing.assert_array_equal(sample["image"][0].numpy() > 0.5, sample["label"][0].numpy())
                np.testing.assert_array_equal(sample["image"][1].numpy() > 0.5, sample["label"][0].numpy())
                np.testing.assert_allclose(sample["image"].affine.numpy(), sample["label"].affine.numpy())

    def test_geometry_scaling_channels_and_dataloader(self):
        array = np.arange(5 * 6 * 7 * 2, dtype=np.int16).reshape(5, 6, 7, 2)
        affine = np.array([[0, -2, 0.2, 30], [1, 0, 0, 40], [0, 0, 3, -10], [0, 0, 0, 1]], dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mla"
            image(array, affine, scale=2, offset=-5).save(path)
            loaded = LoadImage(ensure_channel_first=True, dtype=None)(path)
            np.testing.assert_array_equal(loaded.numpy(), np.moveaxis(array.astype(float) * 2 - 5, -1, 0))
            np.testing.assert_allclose(loaded.affine.numpy(), affine)
            self.assertEqual(tuple(loaded.meta["spatial_shape"]), (5, 6, 7))
            for context in ("fork", "spawn"):
                loader = DataLoader(
                    Dataset([str(path)] * 4, transform=LoadImage()),
                    num_workers=2,
                    batch_size=2,
                    multiprocessing_context=context,
                )
                for batch in loader:
                    np.testing.assert_array_equal(batch[0].numpy(), array * 2 - 5)

    def test_lps_and_reordered_matrix_columns(self):
        array = np.arange(120, dtype=np.uint8).reshape(4, 5, 6)
        # Matrix columns are z,x,y; rows are L,P,S.
        matrix = np.array([[0, -1, 0, -10], [0, 0, -2, -20], [3, 0, 0, 30], [0, 0, 0, 1]], dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lps.mla"
            image(array, matrix, ("left", "posterior", "superior"), ("z", "x", "y")).save(path)
            result = LoadImage(dtype=None)(path)
            np.testing.assert_array_equal(result.numpy(), array)
            np.testing.assert_allclose(
                result.affine.numpy(), [[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]]
            )

    def test_stacking_and_mismatched_geometry(self):
        array = np.zeros((3, 4, 5), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.mla" for i in range(2)]
            for p in paths:
                image(array, np.eye(4)).save(p)
            self.assertEqual(tuple(LoadImage(dtype=None)(paths).shape), (2, 3, 4, 5))
            moved = np.eye(4)
            moved[0, 3] = 1
            image(array, moved).save(paths[1])
            reader = MLArrayReader()
            with self.assertRaises(RuntimeError):
                reader.get_data(reader.read(paths))

    def test_invalid_geometry_and_suffix(self):
        reader = MLArrayReader()
        self.assertFalse(reader.verify_suffix("x.nii.gz"))
        self.assertTrue(reader.verify_suffix("x.mla"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.mla"
            mla.MLArray(np.zeros((3, 4, 5))).save(path)
            with self.assertRaises(ValueError):
                reader.get_data(reader.read(path))
            path.write_bytes(b"corrupt")
            with self.assertRaises((RuntimeError, ValueError, OSError)):
                reader.read(path)


class TestMissingMLArray(unittest.TestCase):
    def test_optional_dependency_does_not_break_numpy_loading(self):
        code = """
import importlib.abc, sys, tempfile
from pathlib import Path
import numpy as np
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlarray' or fullname.startswith('mlarray.'):
            raise ModuleNotFoundError('mlarray deliberately unavailable')
sys.meta_path.insert(0, Block())
from monai.data import MLArrayReader
from monai.transforms import LoadImage
from monai.utils import OptionalImportError
try:
    MLArrayReader()
except OptionalImportError:
    pass
else:
    raise AssertionError('missing dependency must raise OptionalImportError')
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'test.npy'
    np.save(path, np.zeros((2,3,4)))
    assert tuple(LoadImage()(path).shape) == (2,3,4)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
