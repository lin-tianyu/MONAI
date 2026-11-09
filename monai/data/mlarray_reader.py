# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np

from monai.data.image_reader import ImageReader, _copy_compatible_dict, _stack_images
from monai.data.utils import is_supported_format
from monai.utils import MetaKeys, SpaceKeys, ensure_tuple, optional_import, require_pkg

__all__ = ["MLArrayReader"]


@require_pkg(pkg_name="mlarray", version="1.0.0")
class MLArrayReader(ImageReader):
    """Read MLArray v1 medical images with three spatial axes and an optional channel axis.

    Requires ``pip install 'monai[mlarray]'``. Array indexing is preserved; the selected
    physical coordinate system is converted to RAS in millimeters. Non-anatomical
    spaces, time/component axes, and arrays without spatial geometry are rejected.
    Value scale and offset are applied, matching NIfTI reader intensity semantics.

    Args:
        channel_dim: validate the declared channel axis, or ``"no_channel"``.
        space: name of the physical space; None selects the file's first space.

    ``read`` returns MLArray objects opened read-only. ``get_data`` consumes and closes
    these objects, including on failure. Callers using ``read`` alone must close them.
    ``LoadImage`` materializes the full array; mmap alone does not enable patch loading.
    """

    def __init__(self, channel_dim: int | str | None = None, space: str | None = None):
        self.channel_dim = channel_dim
        self.space = space

    def verify_suffix(self, filename):
        return is_supported_format(filename, ("mla",))

    def read(self, data, **kwargs):
        if kwargs:
            raise TypeError("MLArrayReader opens files read-only and accepts no read kwargs.")
        mla, _ = optional_import("mlarray", version="1.0.0")
        images = []
        try:
            for path in ensure_tuple(data):
                images.append(mla.MLArray.open(str(path), mode="r", mmap_mode="r"))
        except Exception:
            for image in images:
                image.close()
            raise
        return images if len(images) > 1 else images[0]

    def get_data(self, img):
        arrays, compatible = [], {}
        images = img if isinstance(img, (list, tuple)) else (img,)
        try:
            for image in images:
                axes = image.axes
                spatial = [i for i, axis in enumerate(axes) if axis.type == "space"]
                channels = [i for i, axis in enumerate(axes) if axis.type == "channel"]
                if len(spatial) != 3 or any(a.type not in ("space", "channel") for a in axes):
                    raise ValueError("MLArrayReader requires three spatial axes and at most one channel axis.")
                if len(channels) > 1:
                    raise ValueError("Multiple channel axes are not supported.")
                channel = channels[0] if channels else None
                if self.channel_dim is not None:
                    if self.channel_dim == "no_channel":
                        if channels:
                            raise ValueError("no_channel conflicts with the declared channel axis.")
                    elif not isinstance(self.channel_dim, int) or not -len(axes) <= self.channel_dim < len(axes):
                        raise ValueError("channel_dim is outside the array dimensions.")
                    elif self.channel_dim % len(axes) != channel:
                        raise ValueError("channel_dim conflicts with the declared spatial/channel axes.")
                space = image.space(self.space)
                if space is None or len(space.axes) != 3:
                    raise ValueError("A three-dimensional anatomical space is required.")
                names = [axes[i].name for i in spatial]
                if set(space.from_array.array_axes) != set(names):
                    raise ValueError("Space must map all three spatial array axes.")
                matrix = space.matrix()
                columns = [list(space.from_array.array_axes).index(name) for name in names] + [3]
                matrix = matrix[:, columns]
                affine = np.eye(4, dtype=np.float64)
                directions = {
                    "right": (0, 1),
                    "left": (0, -1),
                    "anterior": (1, 1),
                    "posterior": (1, -1),
                    "superior": (2, 1),
                    "inferior": (2, -1),
                }
                units = {
                    "millimeter": 1.0,
                    "mm": 1.0,
                    "meter": 1000.0,
                    "m": 1000.0,
                    "micrometer": 0.001,
                    "um": 0.001,
                    "centimeter": 10.0,
                    "cm": 10.0,
                }
                seen = set()
                for row, axis in enumerate(space.axes):
                    if axis.increases_toward not in directions:
                        raise ValueError("Space axes must declare anatomical increases_toward directions.")
                    dst, sign = directions[axis.increases_toward]
                    if dst in seen:
                        raise ValueError("Space axes do not form an anatomical coordinate system.")
                    seen.add(dst)
                    unit = axis.unit_code or axis.unit
                    if unit not in units:
                        raise ValueError(f"Unsupported spatial unit: {unit!r}.")
                    affine[dst] = matrix[row] * sign * units[unit]
                array = image.to_numpy()
                scale = 1.0 if image.values.scale is None else image.values.scale
                offset = 0.0 if image.values.offset is None else image.values.offset
                if scale != 1 or offset != 0:
                    array = array.astype(np.float64) * scale + offset
                header = {
                    MetaKeys.AFFINE: affine,
                    MetaKeys.ORIGINAL_AFFINE: affine.copy(),
                    MetaKeys.SPATIAL_SHAPE: np.asarray([image.shape[i] for i in spatial]),
                    MetaKeys.ORIGINAL_CHANNEL_DIM: float("nan") if channel is None else channel,
                    MetaKeys.SPACE: SpaceKeys.RAS,
                }
                _copy_compatible_dict(header, compatible)
                arrays.append(array)
            return _stack_images(arrays, compatible), compatible
        finally:
            for image in images:
                image.close()
