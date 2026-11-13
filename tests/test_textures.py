# SPDX-License-Identifier: MIT
"""The procedural textures must stay pixel-identical to the images of the original script."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from pipe_rover import textures


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _digest(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()[:16]


# SHA-256 (first 16 hex digits) of the RGB pixels that the simulator uses with its default arguments:
# ground_mud (1024^2, seed 99), sky_overcast (2048x1024, seed 88), pipe corrosion (512^2, seed 42).
GOLDEN = [
    ("generate_corrosion_texture", {"size": 512, "seed": 42}, "e13454382d5407d0"),
    ("generate_mud_texture", {"size": 1024, "seed": 99}, "ba917d5cd56254a6"),
    ("generate_sky_texture", {"width": 2048, "height": 1024, "seed": 88}, "a0212d3cf840a1e4"),
]


@pytest.mark.parametrize(("name", "kwargs", "digest"), GOLDEN, ids=[g[0] for g in GOLDEN])
def test_default_scene_textures_are_reproduced_exactly(tmp_path, name, kwargs, digest):
    out = tmp_path / "tex.png"
    assert getattr(textures, name)(str(out), **kwargs) == str(out)
    assert _digest(_pixels(out)) == digest


SMALL = [
    ("generate_corrosion_texture", {"size": 96}, (96, 96, 3)),
    ("generate_mud_texture", {"size": 96}, (96, 96, 3)),
    ("generate_sky_texture", {"width": 128, "height": 64}, (64, 128, 3)),
]


@pytest.mark.parametrize(("name", "kwargs", "shape"), SMALL, ids=[s[0] for s in SMALL])
def test_shape_dtype_and_seed_behaviour(tmp_path, name, kwargs, shape):
    gen = getattr(textures, name)
    a, b, c = (tmp_path / f"{n}.png" for n in "abc")
    gen(str(a), seed=1, **kwargs)
    gen(str(b), seed=1, **kwargs)
    gen(str(c), seed=2, **kwargs)
    pa, pb, pc = _pixels(a), _pixels(b), _pixels(c)
    assert pa.shape == shape and pa.dtype == np.uint8
    assert np.array_equal(pa, pb), "same seed must give the same image"
    assert not np.array_equal(pa, pc), "different seeds must give different images"
