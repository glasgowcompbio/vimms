import bisect
import itertools
import math
import random
import csv
from abc import abstractmethod
from collections import OrderedDict

from PIL import ImageColor
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Common import path_or_mzml
from vimms.Box import GenericBox


class RGBAColour():
    def __init__(self, R, G, B, A=1.0): 
        self.R, self.G, self.B, self.A = R, G, B, A
        
    @classmethod
    def from_hexcode(cls, hexcode, A=1.0):
        R, G, B = ImageColor.getcolor(hexcode, "RGB")
        return RGBAColour(R, G, B, A)
        
    def to_hexcode(self):
        return f"#{self.R:02x}{self.G:02x}{self.B:02x}"

    def __repr__(self):
        return f"RGBAColour(R={self.R}, G={self.G}, B={self.B}, A={self.A})"

    def __add__(self, other): 
        return RGBAColour(
            self.R + other.R,
            self.G + other.G,
            self.B + other.B,
            self.A + other.A
        )

    def __mul__(self, scalar): 
        return RGBAColour(
            self.R * scalar,
            self.G * scalar,
            self.B * scalar,
            self.A * scalar
        )

    def __rmul__(self, scalar): 
        return self.__mul__(self.scalar)

    def squash(self): 
        return (
            self.R / 255.0, 
            self.G / 255.0, 
            self.B / 255.0, 
            self.A
        )

    def correct_bounds(self):
        self.R = max(0, min(255, self.R))
        self.G = max(0, min(255, self.G))
        self.B = max(0, min(255, self.B))
        self.A = max(0.0, min(1.0, self.A))

    def interpolate(self, others, weights=None):
        colours = [self] + others
        weights = [1 / len(colours) for _ in
                   colours] if weights is None else weights
        start = RGBAColour(0, 0, 0, 0.0)
        new_c = sum((c * w for c, w in zip(colours, weights)), start)
        new_c.correct_bounds()
        return new_c
        

class ColourMap():
    PURE_RED = RGBAColour(255, 0, 0)
    PURE_GREEN = RGBAColour(0, 255, 0)
    PURE_BLUE = RGBAColour(0, 0, 255)

    RED = RGBAColour(237, 28, 36)
    ORANGE = RGBAColour(255, 127, 39)
    YELLOW = RGBAColour(255, 242, 0)
    GREEN = RGBAColour(34, 177, 76)
    LIGHT_BLUE = RGBAColour(0, 162, 232)
    INDIGO = RGBAColour(63, 72, 204)
    VIOLET = RGBAColour(163, 73, 164)

    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def assign_colours(self, boxes, key): pass

    def add_to_subplot(self, ax, boxes, key):
        for box, colour in self.assign_colours(boxes, key):
            ax.add_patch(
                patches.Rectangle(
                    (box.pt1.x, box.pt1.y),
                    box.pt2.x - box.pt1.x,
                    box.pt2.y - box.pt1.y,
                    linewidth=1, ec="black", fc=colour.squash()))

    def get_plot(self, boxes, key):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes, key)
        ax.set_xlim([min(b.pt1.x for b in boxes), max(b.pt2.x for b in boxes)])
        ax.set_ylim([min(b.pt1.y for b in boxes), max(b.pt2.y for b in boxes)])
        return plt


class FixedMap(ColourMap):
    """Allows specifying a mapping from an enumeration of some property of
    interest to some list of colours."""

    def __init__(self, mapping):
        self.mapping = mapping

    def assign_colours(self, boxes, key):
        return ((b, self.mapping[min(key(b), len(self.mapping) - 1)]) for b in
                boxes)

    def unique_colours(self, boxes):
        return ((b, self.mapping[min(i, len(self.mapping) - 1)]) for i, b in
                enumerate(boxes))


class InterpolationMap(ColourMap):
    """Allows assigning colours by interpolating between pairs of n colours
    specified across a value range of some property of interest."""

    def __init__(self, colours):
        self.colours = list(colours)

    def assign_colours(self, boxes, key, minm=None, maxm=None):
        minm = min(key(b) for b in boxes) if minm is None else minm
        maxm = max(key(b) for b in boxes) if maxm is None else maxm
        intervals = [
            minm + i * (maxm - minm) / (len(self.colours) - 1)
            for i in range(len(self.colours))
        ]

        def get_colour(box):
            if (math.isclose(minm, maxm)):
                return self.colours[0]
            elif (key(box) < intervals[0]):
                return self.colours[0]
            elif (key(box) > intervals[-1]):
                return self.colours[-1]
            else:
                i = next(
                    i - 1 for i, threshold in enumerate(intervals) 
                    if threshold >= key(box)
                )
                weight = (key(box) - intervals[i]) / (intervals[i + 1] - intervals[i])
                return self.colours[i].interpolate(
                    [self.colours[i + 1]], 
                    weights=[1 - weight, weight]
                )

        return ((b, get_colour(b)) for b in boxes)


class AutoColourMap(ColourMap):

    def __init__(self, colour_picker, reuse_colours=False):
        self.colour_picker = colour_picker
        self.reuse_colours = reuse_colours

    @staticmethod
    def random_colours(boxes):
        return ((b, RGBAColour(*(random.uniform(0, 255) for _ in range(3))))
                for b in boxes)

    def assign_colours(self, boxes):
        # if there's no path from one box to another when we build a graph of
        # their overlaps, we can re-use colours
        pairs = [(top, set()) for b in boxes for top in b.parents]
        top_level = OrderedDict(pairs)  # need uniqueness and maintain ordering
        if (self.reuse_colours):
            for b in boxes:
                for top in b.parents:
                    top_level[top].add(
                        b)  # note same set references in pairs and top_level

            components, indices = [], [-1 for _ in pairs]
            for i, (parent, children) in enumerate(top_level.items()):
                if (indices[i] == -1):
                    indices[i] = len(components)
                    components.append(OrderedDict([(parent, None)]))
                update = [(j, k) for j, (k, v) in enumerate(pairs[i:]) if
                          children & v]
                for (j, k) in update:
                    indices[j] = indices[i]
                    components[indices[i]][k] = None
        else:
            components = [top_level]

        top_level_colours = {top: colour for cs in components for
                             top, colour in self.colour_picker(cs.keys())}

        def interpolate_lower(box):
            return top_level_colours[box.parents[0]].interpolate(
                [top_level_colours[b] for b in box.parents[1:]])

        return ((b, interpolate_lower(b)) for b in boxes)

    def add_to_subplot(self, ax, boxes):
        for box, colour in self.assign_colours(boxes):
            ax.add_patch(
                patches.Rectangle((box.pt1.x, box.pt1.y),
                                  box.pt2.x - box.pt1.x,
                                  box.pt2.y - box.pt1.y, linewidth=1,
                                  ec="black", fc=colour.squash()))

    def get_plot(self, boxes):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes)
        ax.set_xlim([min(b.pt1.x for b in boxes), max(b.pt2.x for b in boxes)])
        ax.set_ylim([min(b.pt1.y for b in boxes), max(b.pt2.y for b in boxes)])
        return plt

class PlotPoints():
    def __init__(self, ms1_points, ms2s=None, markers={}):
        self.ms1_points, self.ms2s = ms1_points, ms2s
        self.markers = markers
        self.active_ms1 = np.ones((ms1_points.shape[0]))
        self.active_ms2 = np.ones((ms2s.shape[0]))

    @classmethod
    def from_mzml(cls, mzml):
        mzml = path_or_mzml(mzml)
        scan_dict = {1: [], 2: []}
        for s in mzml.scans:
            if (s.ms_level in scan_dict):
                scan_dict[s.ms_level].append(s)
        scan_dict[1] = sorted(scan_dict[1], key=lambda s: s.rt_in_seconds)
        ms1_points = np.array(
            [[s.rt_in_seconds, mz, intensity] for s in scan_dict[1] for
             mz, intensity in s.peaks])
        ms2s = np.array(
            [[s.rt_in_seconds, s.precursor_mz, s.get_max_intensity().intensity]
             for s in scan_dict[2]])
        return cls(ms1_points, ms2s=ms2s)

    def bound_points(self, pts, min_rt=None, max_rt=None, min_mz=None,
                     max_mz=None):
        all_true = np.array(np.ones_like(pts[:, 0]), dtype=np.bool)
        select_rt = (all_true if (min_rt is None) else (
            pts[:, 0] >= min_rt)) & (
            all_true if (max_rt is None) else (pts[:, 0] <= max_rt))
        select_mz = (all_true if (min_mz is None) else (
            pts[:, 1] >= min_mz)) & (
            all_true if (max_mz is None) else (pts[:, 1] <= max_mz))
        return (select_rt & select_mz)

    def get_points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None,
                             max_mz=None):
        active_ms1 = self.bound_points(self.ms1_points, min_rt=min_rt,
                                       max_rt=max_rt, min_mz=min_mz,
                                       max_mz=max_mz) if len(
            self.ms1_points) > 0 else np.array([[]])
        active_ms2 = self.bound_points(self.ms2s, min_rt=min_rt, max_rt=max_rt,
                                       min_mz=min_mz, max_mz=max_mz) if len(
            self.ms2s) > 0 else np.array([[]])
        return active_ms1, active_ms2

    def points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None,
                         max_mz=None):
        self.active_ms1, self.active_ms2 = self.get_points_in_bounds(
            min_rt=min_rt, max_rt=max_rt, min_mz=min_mz,
            max_mz=max_mz)

    def mark_precursors(self, max_error=10):
        markers = ["o" for _ in enumerate(self.ms1_points)]
        ms1_min, ms1_max = 0, 0
        for rt, precursor_mz, _ in self.ms2s:
            ms1_max = bisect.bisect_right(self.ms1_points[:, 0], rt)
            precursor_time = self.ms1_points[ms1_max - 1, 0]
            ms1_min = bisect.bisect_left(self.ms1_points[:, 0], precursor_time)
            if (ms1_max - ms1_min > 0):
                i = np.argmin(np.abs(self.ms1_points[ms1_min:ms1_max,
                                     1] - precursor_mz)) + ms1_min
                if (1e6 * np.abs(self.ms1_points[
                        i, 1] - precursor_mz) / precursor_mz <= max_error):
                    markers[i] = "x"
        markers = np.array(markers)[self.active_ms1]
        marker_dict = {m: [] for m in set(markers)}
        for i, m in enumerate(markers):
            marker_dict[m].append(i)
        self.markers = marker_dict

    def colour_by_intensity(self, originals, intensities, abs_scaling):
        cmap = InterpolationMap(
            [RGBAColour(238, 238, 238), ColourMap.YELLOW, ColourMap.RED,
             ColourMap.PURE_BLUE])
        if (abs_scaling):
            minm, maxm = math.log(np.min(self.ms1_points[:, 2])), math.log(
                np.max(self.ms1_points[:, 2]))
            colours = np.array(list(
                cmap.assign_colours(intensities, lambda x: math.log(x),
                                    minm=minm, maxm=maxm)))
        else:
            colours = np.array(
                list(cmap.assign_colours(intensities, lambda x: math.log(x))))
        return colours

    def plot_ms1s(self, ax, min_rt=None, max_rt=None, min_mz=None, max_mz=None,
                  abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz,
                              max_mz=max_mz)
        rts, mzs, intensities = self.ms1_points[self.active_ms1, 0], \
            self.ms1_points[self.active_ms1, 1], \
            self.ms1_points[self.active_ms1, 2]
        colours = self.colour_by_intensity(self.ms1_points, intensities,
                                           abs_scaling=abs_scaling)
        self.mark_precursors()
        for marker, idxes in self.markers.items():
            ax.scatter(rts[idxes], mzs[idxes],
                       color=[colour.squash() for _, colour in colours[idxes]],
                       marker=marker)

    def plot_ms2s(self, ax, min_rt=None, max_rt=None, min_mz=None, max_mz=None,
                  abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz,
                              max_mz=max_mz)
        rts, mzs, intensities = self.ms2s[self.active_ms2, 0], self.ms2s[
            self.active_ms2, 1], self.ms2s[
            self.active_ms2, 2]
        colours = self.colour_by_intensity(self.ms2s, intensities,
                                           abs_scaling=abs_scaling)
        ax.scatter(rts, mzs, color=[colour.squash() for _, colour in colours],
                   marker="x")


class PlotBox():
    RT_TOLERANCE, MZ_TOLERANCE = 0.4, 0.4

    def __init__(self, min_rt, max_rt, min_mz, max_mz, intensity, ec="black",
                 fc=[0, 0, 0, 0]):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.min_mz, self.max_mz = min_mz, max_mz
        self.intensity = intensity
        self.ec, self.fc = ec, fc

    def __repr__(self):
        return f"PlotBox(min_rt={self.min_rt}, max_rt={self.max_rt}, " \
               f"min_mz={self.min_mz}, max_mz={self.max_mz}, " \
               f"apex_intensity={self.intensity})"

    @classmethod
    def from_box(cls, box):
        return cls(box.pt1.x, box.pt2.x, box.pt1.y, box.pt2.y, box.intensity)

    @classmethod
    def from_roi(cls, roi, min_rt_width, min_mz_width):
        return cls.from_box(roi.to_box(min_rt_width, min_mz_width))

    @classmethod
    def from_env(cls, env, min_rt_width, min_mz_width):
        roi_builder = env.controller.roi_builder
        live_roi = roi_builder.live_roi
        dead_roi = roi_builder.dead_roi
        junk_roi = roi_builder.junk_roi
        return [cls.from_roi(roi, min_rt_width, min_mz_width) for roi in
                itertools.chain(live_roi, dead_roi, junk_roi)]

    @classmethod
    def from_grid(cls, grid):
        return [cls.from_box(b) for b in grid.all_boxes()]

    @classmethod
    def from_roi_aligner(cls, aligner, ps=None):
        if (ps is None):
            pses = aligner.peaksets
        else:
            pses = [ps]

        plot_boxes = []
        for ps in pses:
            ps_boxes = []
            for box in aligner.peaksets2boxes[ps]:
                for p in ps.peaks:
                    rt_range, mz_range = box.rt_range_in_seconds, box.mz_range
                    pb = cls(rt_range[0], rt_range[1], mz_range[0],
                             mz_range[1], p.intensity)
                    ps_boxes.append(pb)
            plot_boxes.append(ps_boxes)

        return plot_boxes
        
    def serialise_info(self):
        return [self.min_rt, self.max_rt, self.min_mz, self.max_mz, self.intensity]

    def box_in_bounds(self, min_rt=None, max_rt=None, min_mz=None,
                      max_mz=None):
        return (
            (min_rt is None or self.max_rt >= min_rt)
            and (max_rt is None or self.min_rt <= max_rt)
            and
                (
                    (min_mz is None or self.max_mz >= min_mz)
                    and (max_mz is None or self.min_mz <= max_mz)
            )
        )

    def add_to_plot(self, ax):
        x1, y1 = self.min_rt, self.min_mz
        xlen, ylen = (self.max_rt - self.min_rt), (self.max_mz - self.min_mz)
        ax.add_patch(
            patches.Rectangle((x1, y1), xlen, ylen, linewidth=1, ec=self.ec,
                              fc=self.fc))

    def get_plot_bounds(self, rt_buffer=None, mz_buffer=None):
        if (rt_buffer is None):
            rt_buffer = (self.max_rt - self.min_rt) * self.RT_TOLERANCE
        if (mz_buffer is None):
            mz_buffer = (self.max_mz - self.min_mz) * self.MZ_TOLERANCE
        xbounds = [self.min_rt - rt_buffer, self.max_rt + rt_buffer]
        ybounds = [self.min_mz - mz_buffer, self.max_mz + mz_buffer]
        return xbounds, ybounds

    def plot_box(self, ax, mzml, abs_scaling=False, other_boxes=[],
                 rt_buffer=None, mz_buffer=None):
        xbounds, ybounds = self.get_plot_bounds(rt_buffer=rt_buffer,
                                                mz_buffer=mz_buffer)
        pts = PlotPoints.from_mzml(mzml)
        pts.plot_ms1s(ax, min_rt=xbounds[0], max_rt=xbounds[1],
                      min_mz=ybounds[0], max_mz=ybounds[1],
                      abs_scaling=abs_scaling)
        self.add_to_plot(ax)
        for b in other_boxes:
            if (b.box_in_bounds(min_rt=xbounds[0], max_rt=xbounds[1],
                                min_mz=ybounds[0], max_mz=ybounds[1])):
                b.add_to_plot(ax)
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        ax.set(xlabel="RT (Seconds)", ylabel="m/z")


def boxes2csv(fname, boxes, colours=None):
    with open(fname, "w") as f:
        w = csv.writer(f)
        headers = ["rtLo", "rtHi", "mzLo", "mzHi", "intensity"]
        if(colours is None): 
            colours = itertools.repeat(None)
        else:
            headers.extend(["color", "opacity"])
        w.writerow(headers)
        
        for b, c in zip(boxes, colours):
            ls = b.serialise_info()
            if(not c is None):
                ls.extend([c.to_hexcode(), c.A])
            w.writerow(ls)

    
def csv2boxes(fname):
    with open(fname, "r") as f:
        r = csv.DictReader(f)
        
        boxes, colours = [], []
        for row in r:
            boxes.append(
                GenericBox(
                    row["rtLo"],
                    row["rtHi"],
                    row["mzLo"],
                    row["mzHi"]
                )
            )
            
            if("color" in row):
                A = float(row.get("opacity", 1.0))
                colours.append(
                    RGBAColour.from_hexcode(row["color"], A=A)
                )
                
    return boxes, colours