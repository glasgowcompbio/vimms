import numpy as np
import pytest

from vimms.Box import GenericBox, DictGrid, ArrayGrid
from vimms.BoxVisualise import RGBAColour, FixedMap, InterpolationMap


def test_generic_box_overlap_and_area():
    b1 = GenericBox(0, 10, 0, 10)
    b2 = GenericBox(5, 15, 5, 15)
    assert b1.area() == 100
    assert b1.overlaps_with_box(b2)
    assert pytest.approx(b1.overlap_raw(b2)) == 25
    union = b1.area() + b2.area() - 25
    assert pytest.approx(b1.overlap_2(b2)) == 25 / union
    assert pytest.approx(b1.overlap_3(b2)) == 0.25


def test_generic_box_non_overlap_split():
    b1 = GenericBox(0, 10, 0, 10)
    b2 = GenericBox(5, 15, 5, 15)
    splits = b1.non_overlap_split(b2)
    assert len(splits) == 2
    pts = sorted((s.pt1.x, s.pt2.x, s.pt1.y, s.pt2.y) for s in splits)
    assert pts == [(0.0, 5.0, 0.0, 10.0), (5.0, 10.0, 0.0, 5.0)]


def test_generic_box_apply_min_box_ppm_expands():
    b = GenericBox(0, 1, 0, 1)
    expanded = b.apply_min_box_ppm(xwidth=4e6)
    assert expanded.pt1.x < b.pt1.x
    assert expanded.pt2.x > b.pt2.x


def _check_grid(Grid):
    g = Grid(0, 10, 5, 0, 10, 5)
    q = GenericBox(0, 10, 0, 10)
    assert g.approx_non_overlap(q) == 1.0
    b1 = GenericBox(0, 5, 0, 5)
    g.register_box(b1)
    assert np.isclose(g.approx_non_overlap(q), 5 / 9)
    g.register_box(GenericBox(5, 10, 0, 5))
    assert np.isclose(g.approx_non_overlap(q), 3 / 9)


def test_dict_and_array_grid():
    for grid in (DictGrid, ArrayGrid):
        _check_grid(grid)


def test_rgba_colour_operations():
    c = RGBAColour.from_hexcode("#ff0000", A=0.5)
    assert (c.R, c.G, c.B, c.A) == (255, 0, 0, 0.5)
    assert c.to_hexcode() == "#ff0000"

    other = RGBAColour(10, 20, 30, 0.5)
    added = c + other
    assert (added.R, added.G, added.B, added.A) == (265, 20, 30, 1.0)

    scaled = other * 0.5
    assert (scaled.R, scaled.G, scaled.B, scaled.A) == (5, 10, 15, 0.25)

    bad = RGBAColour(300, -10, 260, 2)
    bad.correct_bounds()
    assert (bad.R, bad.G, bad.B, bad.A) == (255, 0, 255, 1.0)

    inter = c.interpolate([RGBAColour(0, 0, 0, 0.0)], weights=[0.5, 0.5])
    assert np.isclose(inter.R, 127.5)
    assert np.isclose(inter.A, 0.25)


def test_fixed_and_interpolation_map():
    boxes = [GenericBox(0, 1, 0, 1), GenericBox(2, 3, 2, 3)]
    fmap = FixedMap([RGBAColour(1, 1, 1), RGBAColour(2, 2, 2)])
    cols = list(fmap.unique_colours(boxes))
    assert cols[0][1].R == 1 and cols[1][1].R == 2

    imap = InterpolationMap([RGBAColour(0, 0, 0), RGBAColour(100, 100, 100)])
    cols = list(imap.assign_colours(boxes, lambda b: b.pt1.x, minm=0, maxm=2))
    assert cols[0][1].R == 0
    assert cols[1][1].R == 100

    mid_box = GenericBox(0.5, 1.5, 0, 1)
    mid_col = list(imap.assign_colours([mid_box], lambda b: b.pt1.x, minm=0, maxm=1))[0][1]
    assert np.isclose(mid_col.R, 50.0)
