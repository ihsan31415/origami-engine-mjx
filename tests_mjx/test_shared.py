"""Tests for shared (backend-independent) functionality."""

import numpy as np
import pytest

from origami_engine_mjx import (
    Crease,
    build_pattern,
    load_creases_from_json,
    load_default_creases,
    normalize_creases,
    save_creases_to_json,
)
from origami_engine_mjx.model import generate_xml


def test_default_pattern_is_valid():
    creases = load_default_creases()
    pat = build_pattern(creases)
    assert pat.valid
    assert pat.n_faces == 16
    assert pat.n_loops == 9


def test_normalize_creases_from_dicts():
    raw = [{"p1": [0.0, 0.0], "p2": [0.5, 0.5], "angle": 0.5}]
    creases = normalize_creases(raw)
    assert len(creases) == 1
    assert isinstance(creases[0], Crease)
    assert creases[0].angle == 0.5


def test_normalize_creases_rejects_empty():
    with pytest.raises(ValueError):
        normalize_creases([])


def test_json_round_trip(tmp_path):
    creases = load_default_creases()
    out = tmp_path / "creases.json"
    save_creases_to_json(creases, out, angle_unit="degree")
    loaded = load_creases_from_json(out)
    assert len(loaded) == len(creases)
    for a, b in zip(loaded, creases):
        assert a.p1 == pytest.approx(b.p1)
        assert a.p2 == pytest.approx(b.p2)
        assert a.angle == pytest.approx(b.angle)


def test_generate_xml_produces_valid_mjcf():
    """generate_xml must not require any MuJoCo runtime — pure string output."""
    creases = load_default_creases()
    pat = build_pattern(creases)
    xml, info = generate_xml(pat, creases)
    assert xml.startswith("<mujoco")
    assert xml.endswith("</mujoco>")
    assert info["n_faces"] == 16
    assert info["n_hinges"] == 15
    assert info["n_loops"] == 9
    assert info["targets"].shape == (15,)
    # Should contain key structural elements.
    assert '<freejoint name="root"/>' in xml
    assert '<connect body1="face_' in xml
    assert '<position name="fold_' in xml
    assert 'name="table"' in xml


def test_invalid_pattern_detected():
    bad = [
        Crease((0.0, 0.0), (0.01, 0.0), 0.0),
        Crease((0.01, 0.0), (0.01, 0.0001), 0.0),
        Crease((0.01, 0.0001), (0.0, 0.0001), 0.0),
    ]
    pat = build_pattern(bad)
    assert not pat.valid
