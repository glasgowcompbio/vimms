import os
import copy
import bisect
import itertools
import math
import random
import csv
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
from PIL import ImageColor
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Common import path_or_mzml, get_scan_times_combined, get_scan_times
from vimms.Box import GenericBox, BoxGrid


class RGBAColour():
    def __init__(self, R, G, B, A=1.0): 
        self.R, self.G, self.B, self.A = R, G, B, A
        
    @classmethod
    def from_hexcode(cls, hexcode, A=1.0):
        R, G, B = ImageColor.getcolor(hexcode, "RGB")
        return RGBAColour(R, G, B, A)
        
    def to_hexcode(self):
        return f"#{self.R:02x}{self.G:02x}{self.B:02x}"
        
    def to_tuple(self):
        return (self.R, self.G, self.B, self.A)
        
    def to_plotly(self):
        return f"rgb({self.R}, {self.G}, {self.B})", self.A

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
                    linewidth=1, 
                    ec="black", 
                    fc=colour.squash()
                )
            )

    def get_plot(self, boxes, key):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes, key)
        ax.set_xlim([min(b.pt1.x for b in boxes), max(b.pt2.x for b in boxes)])
        ax.set_ylim([min(b.pt1.y for b in boxes), max(b.pt2.y for b in boxes)])
        return fig


class FixedMap(ColourMap):
    """Allows specifying a mapping from an enumeration of some property of
    interest to some list of colours."""

    def __init__(self, mapping):
        self.mapping = mapping

    def assign_colours(self, boxes, key):
        return ((b, self.mapping[min(key(b), len(self.mapping) - 1)]) for b in boxes)

    def unique_colours(self, boxes):
        return ((b, self.mapping[min(i, len(self.mapping) - 1)]) for i, b in enumerate(boxes))


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
        if(all(not top.id is None for top, _ in pairs)):
            pairs.sort(key=lambda p: p[0].id)
        top_level = OrderedDict(pairs)  # need uniqueness and maintain ordering
        if(self.reuse_colours):
            for b in boxes:
                for top in b.parents:
                    # note same set references in pairs and top_level
                    top_level[top].add(b)

            components, indices = [], [-1 for _ in pairs]
            for i, (parent, children) in enumerate(top_level.items()):
                if (indices[i] == -1):
                    indices[i] = len(components)
                    components.append(OrderedDict([(parent, None)]))
                update = [
                    (j, k) for j, (k, v) in enumerate(pairs[i:]) 
                    if children & v
                ]
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
        return fig

class PlotPoints():
    hovertemplate = """
    Retention Time: %{x}<br>
    m/z %{y}<br>
    Intensity: %{customdata}
    """

    def __init__(self, ms1_points, ms2s=None, markers={}):
        self.ms1_points = np.array(ms1_points)
        self.ms2s = np.array(ms2s) if not ms2s is None else np.zeros((0, 3))
        self.active_ms1 = np.ones((len(self.ms1_points)), dtype=np.bool)
        self.active_ms2 = np.ones((len(self.ms2s)), dtype=np.bool)
        self.markers = markers

    @classmethod
    def from_mzml(cls, mzml):
        mzml = path_or_mzml(mzml)
        ms1_points, ms2s = [], []
        for s in mzml.scans:
            if(s.ms_level == 1):
                ms1_points.extend(
                    [[s.rt_in_seconds, mz, intensity] for mz, intensity in s.peaks]
                )

            if(s.ms_level == 2):
                ms2s.append(
                    [s.rt_in_seconds, s.precursor_mz, s.get_max_intensity().intensity]
                )
                
        ms1_points.sort(key=lambda s: s[0])
        ms2s.sort(key=lambda s: s[0])
        
        return cls(np.array(ms1_points), ms2s=np.array(ms2s))

    def bound_points(self, pts, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        all_true = np.array(np.ones_like(pts[:, 0]), dtype=np.bool)
        select_rt = (
            (all_true if (min_rt is None) else (pts[:, 0] >= min_rt))
            & (all_true if (max_rt is None) else (pts[:, 0] <= max_rt))
        )
        select_mz = (
            (all_true if (min_mz is None) else (pts[:, 1] >= min_mz)) 
            & (all_true if (max_mz is None) else (pts[:, 1] <= max_mz))
        )
        return (select_rt & select_mz)

    def get_points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        
        active_ms1 = self.bound_points(
            self.ms1_points, 
            min_rt=min_rt,
            max_rt=max_rt, 
            min_mz=min_mz,
            max_mz=max_mz
        ) if len(self.ms1_points) > 0 else np.ones((0), dtype=np.bool)
        
        active_ms2 = self.bound_points(
            self.ms2s, 
            min_rt=min_rt, 
            max_rt=max_rt,
            min_mz=min_mz, 
            max_mz=max_mz
        ) if len(self.ms2s) > 0 else np.ones((0), dtype=np.bool)
        
        return active_ms1, active_ms2

    def points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        self.active_ms1, self.active_ms2 = (
            self.get_points_in_bounds(
                min_rt=min_rt, 
                max_rt=max_rt, 
                min_mz=min_mz,
                max_mz=max_mz
            )
        )

    def mark_precursors(self, max_error=10):
        markers = ["0" for _ in enumerate(self.ms1_points)]
        ms1_min, ms1_max = 0, 0
        sorted_ms1s = self.ms1_points
        for rt, precursor_mz, _ in self.ms2s:
            ms1_max = bisect.bisect_right(sorted_ms1s[:, 0], rt)
            precursor_time = sorted_ms1s[ms1_max - 1, 0]
            ms1_min = bisect.bisect_left(sorted_ms1s[:, 0], precursor_time)
            if(ms1_max - ms1_min > 0):
                i = np.argmin(
                    np.abs(sorted_ms1s[ms1_min:ms1_max, 1] - precursor_mz)
                ) + ms1_min
                
                if(1e6 * np.abs(sorted_ms1s[i, 1] - precursor_mz) / precursor_mz <= max_error):
                    markers[i] = "x"
        self.markers = markers
        
    def colour_by_intensity(self, intensities, colour_minm, abs_scaling):
        if(len(intensities) == 0): return np.array([])
        cmap = InterpolationMap([
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED,
            ColourMap.PURE_BLUE
        ])
        
        if(abs_scaling):
            if(not colour_minm is None): minm = colour_minm
            else: minm = math.log(np.min(self.ms1_points[:, 2]))
            maxm = math.log(np.max(self.ms1_points[:, 2]))
            colours = np.array(list(
                cmap.assign_colours(intensities, lambda x: math.log(x), minm=minm, maxm=maxm)
            ))
        else:
            colours = np.array(list(
                cmap.assign_colours(intensities, lambda x: math.log(x), minm=colour_minm)
            ))
        
        return colours

    def mpl_add_ms1s(self, ax, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None,
                    draw_minm=0.0, colour_minm=None,
                    marker_styles=None,
                    abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        if(np.sum(self.active_ms1) == 0): return
        self.mark_precursors()
        
        active_ms1 = self.ms1_points[self.active_ms1, :]
        sort_idx = np.argsort(active_ms1[:, 2])
        srted = active_ms1[sort_idx]
        filter_idx = srted[:, 2] >= draw_minm
        rts, mzs, intensities = srted[filter_idx].T
        
        markers = np.array(self.markers)[self.active_ms1][sort_idx][filter_idx]
        if(marker_styles is None):
            marker_styles = ["o", "x"]
        markers = [marker_styles[0] if m == "0" else marker_styles[1] for m in markers]
        
        colours = self.colour_by_intensity(intensities, colour_minm, abs_scaling)
        for marker in ["o", "x"]:
            idxes = [i for i, m in enumerate(markers) if m == marker]
            ax.scatter(
                rts[idxes], mzs[idxes],
                color=[colour.squash() for _, colour in colours[idxes]],
                marker=marker
            )

    def mpl_add_ms2s(self, ax, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None, 
                    colour_minm=None,
                    abs_scaling=False):

        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        if(np.sum(self.active_ms2) == 0): return
        active_ms2 = self.ms2s[self.active_ms2, :]
        rts, mzs, intensities = active_ms2[np.argsort(active_ms2[:, 2])].T
        colours = self.colour_by_intensity(intensities, colour_minm, abs_scaling)
        ax.scatter(rts, mzs, color=[colour.squash() for _, colour in colours], marker="x")

    def plotly_ms1s(self, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None, 
                    draw_minm=0.0, colour_minm=None,
                    show_precursors=False,
                    abs_scaling=False):
                    
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        if(np.sum(self.active_ms1) == 0): return
        self.mark_precursors()
        
        active_ms1 = self.ms1_points[self.active_ms1, :]
        sort_idx = np.argsort(active_ms1[:, 2])
        srted = active_ms1[sort_idx]
        filter_idx = srted[:, 2] >= draw_minm
        rts, mzs, intensities = active_ms1[filter_idx].T
        
        lit = [math.log(it) for it in intensities]
        colourscale = [
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED, 
            ColourMap.PURE_BLUE
        ]
        
        if(show_precursors):
            markers = np.array(self.markers)[self.ms1_points[:, 2] >= draw_minm][sort_idx]
        else:
            markers = "0"
            
        if(not colour_minm is None):
            cmin = colour_minm
        elif(not abs_scaling is None):
            cmin = min(np.log(self.ms2_points[:, 2]))
        else:
            cmin = None
        
        return go.Scattergl(
                x=rts, 
                y=mzs,
                mode="markers",
                marker=dict(
                    symbol=markers,
                    color=lit,
                    colorscale=[c.to_plotly()[0] for c in colourscale],
                    cmin=cmin,
                    cmax=max(np.log(self.ms1_points[:, 2])) if abs_scaling else max(lit)
                ),
                hovertemplate=self.hovertemplate,
                customdata=intensities
        )

    def plotly_ms2s(self, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None,
                    colour_minm=None,
                    abs_scaling=False):
                    
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        if(np.sum(self.active_ms2) == 0): return
        active_ms2 = self.ms2s[self.active_ms2, :]
        rts, mzs, intensities = active_ms2[np.argsort(active_ms2[:, 2])].T
        lit = [math.log(it) for it in intensities]
        colourscale = [
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED, 
            ColourMap.PURE_BLUE
        ]
        
        if(not colour_minm is None):
            cmin = colour_minm
        elif(not abs_scaling is None):
            cmin = min(np.log(self.ms2_points[:, 2]))
        else:
            cmin = None
        
        return go.Scattergl(
                x=rts, 
                y=mzs,
                mode="markers",
                marker=dict(
                    symbol="x",
                    color=lit,
                    colorscale=[c.to_plotly()[0] for c in colourscale],
                    cmin=cmin,
                    cmax=max(np.log(self.ms1_points[:, 2])) if abs_scaling else max(lit)
                ),
                hovertemplate=self.hovertemplate,
                customdata=intensities
        )


class PlotBox():
    RT_TOLERANCE, MZ_TOLERANCE = 0.4, 0.4

    def __init__(self, min_rt, max_rt, min_mz, max_mz, intensity, ec=RGBAColour(0, 0, 0, 1),
                 fc=RGBAColour(0, 0, 0, 0)):
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
        
    def to_box(self, min_rt_width=0, min_mz_width=0, rt_shift=0, mz_shift=0):
        return GenericBox(
            self.min_rt + rt_shift, 
            self.max_rt + rt_shift, 
            self.min_mz + mz_shift,
            self.max_mz + mz_shift,
            min_xwidth=min_rt_width, 
            min_ywidth=min_mz_width, 
            intensity=self.intensity
        )

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
        return [cls.from_box(b) for b in grid.get_all_boxes()]

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
    
    @classmethod
    def from_evaluator(cls, eva, min_intensity=0.0):
        boxes = {"fragmented" : [], "unfragmented" : []}
        partition = eva.partition_chems(min_intensity=min_intensity, aggregate="max")
        groups = [
            ("fragmented", ColourMap.PURE_BLUE),
            ("unfragmented", ColourMap.PURE_RED)
        ]
        
        for name, colour in groups:
            for row in partition[name]:
                new_b = PlotBox.from_box(row[0])
                new_b.ec = colour
                boxes[name].append(new_b)
                
        return boxes
        
    def serialise_info(self, minutes=False):
        timescale = 60 if minutes else 1
        return [
            self.min_rt / timescale, 
            self.max_rt / timescale, 
            self.min_mz, 
            self.max_mz, 
            self.intensity
        ]

    def box_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        return (
            (min_rt is None or self.max_rt >= min_rt)
            and (max_rt is None or self.min_rt <= max_rt)
            and (min_mz is None or self.max_mz >= min_mz)
            and (max_mz is None or self.min_mz <= max_mz)
        )

    def mpl_add_to_plot(self, ax):
        x1, y1 = self.min_rt, self.min_mz
        xlen, ylen = (self.max_rt - self.min_rt), (self.max_mz - self.min_mz)
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), 
                xlen, 
                ylen, 
                linewidth=1, 
                ec=self.ec.squash(), 
                fc=self.fc.squash()
            )
        )

    def get_plot_bounds(self, rt_buffer=None, mz_buffer=None):
        if (rt_buffer is None):
            rt_buffer = (self.max_rt - self.min_rt) * self.RT_TOLERANCE
        if (mz_buffer is None):
            mz_buffer = (self.max_mz - self.min_mz) * self.MZ_TOLERANCE
        xbounds = [self.min_rt - rt_buffer, self.max_rt + rt_buffer]
        ybounds = [self.min_mz - mz_buffer, self.max_mz + mz_buffer]
        return xbounds, ybounds

    def mpl_plot_box(self, 
                     ax, 
                     mzml, 
                     abs_scaling=False, 
                     other_boxes=[],
                     draw_minm=0.0,
                     colour_minm=None,
                     rt_buffer=None, 
                     mz_buffer=None):
        xbounds, ybounds = self.get_plot_bounds(rt_buffer=rt_buffer,
                                                mz_buffer=mz_buffer)
        pts = PlotPoints.from_mzml(mzml)
        pts.mpl_add_ms1s(
            ax, 
            min_rt=xbounds[0], 
            max_rt=xbounds[1],
            min_mz=ybounds[0], 
            max_mz=ybounds[1],
            draw_minm=draw_minm,
            colour_minm=colour_minm,
            abs_scaling=abs_scaling
        )
        self.mpl_add_to_plot(ax)
        for b in other_boxes:
            if (b.box_in_bounds(min_rt=xbounds[0], max_rt=xbounds[1],
                                min_mz=ybounds[0], max_mz=ybounds[1])):
                b.mpl_add_to_plot(ax)
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        ax.set(xlabel="RT (Seconds)", ylabel="m/z")


class EnvPlotPickler():
    """
        Convenient wrapper to pickle env data for later plotting.
    """
    def __init__(self, env):
        self.processing_times = env.controller.processing_times
        
        try:
            self.bm = env.controller.grid
        except AttributeError:
            pass


def mpl_set_axis_style(ax,
                   labelsize=None,
                   title=None,
                   title_y=None,
                   titlesize=None,
                   linewidth=None,
                   markersize=None,
                   legend_kwargs=None):
    
    if(not labelsize is None):
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(labelsize)
    
    if(not title is None):
        ax.set_title(title, y=title_y)
    
    if(not titlesize is None):
        ax.title.set_fontsize(titlesize)
    
    for ln in ax.lines:
        if(not linewidth is None):
            ln.set_linewidth(linewidth)
        
        if(not markersize is None):
            ln.set_markersize(markersize)
    
    if(not legend_kwargs is None):
        ax.legend(**legend_kwargs)


def mpl_set_figure_style(fig,
                     tick_kwargs=None,
                     axis_borderwidth=None,
                     axis_kwargs=None,
                     suptitle=None, 
                     suptitle_y=None,
                     suptitle_size=18,
                     figure_sizes=None):
    
    for ax in fig.axes:
        if(not tick_kwargs is None):
            ax.tick_params(**tick_kwargs)
            
        if(not axis_borderwidth is None):
            for pos in ["top", "bottom", "left", "right"]:
                ax.spines[pos].set_linewidth(axis_borderwidth)
        
        if(not axis_kwargs is None):
            mpl_set_axis_style(ax, **axis_kwargs)
            
    if(not suptitle is None):
        fig.suptitle(suptitle, y=suptitle_y, fontsize=suptitle_size)
    
    if(not figure_sizes is None):
        fig.set_size_inches(*figure_sizes)


def mpl_results_plot(experiment_names, 
                     evals, 
                     min_intensity=0.0,
                     keys=None,
                     colours=None, 
                     markers=None,
                     mode="absolute"):
    
    mode = mode.lower()
    if(mode == "absolute"):
        mode_str = ""
    elif(mode == "relative"):
        mode_str = "(Relative)"
    
    if(keys is None):
        keys = ["cumulative_coverage_proportion", "cumulative_intensity_proportion"]
        
    layouts = {
        "cumulative_coverage_proportion" : {
            "title" : "Cumulative Coverage",
            "ylabel" :  f"Cumulative Coverage Proportion {mode_str}",
        },
        
        "cumulative_intensity_proportion" : {
            "title" : "Cumulative Intensity Proportion",
            "ylabel" : f"Cumulative Intensity Proportion {mode_str}",
        },
        
        "cumulative_covered_intensities_proportion" : {
            "title" : "Cumulative Intensity Proportion (Covered Peaks Only)",
            "ylabel" : f"Cumulative Intensity Proportion {mode_str}",
        }
    }
                     
    fig, axes = plt.subplots(1, len(keys))
    try:
        axes[0]
    except TypeError:
        axes = [axes]
    
    if(colours is None): use_colours = itertools.repeat(None)
    else: use_colours = copy.copy(colours)
    
    if(markers is None): use_markers = itertools.repeat(None)
    else: use_markers = copy.copy(markers)
    
    results_list = [eva.evaluation_report(min_intensity=min_intensity) for eva in evals]
    for key, ax in zip(keys, axes):
        itr = zip(experiment_names, results_list, use_colours, use_markers)
        if(mode == "relative"):
            means = np.mean(
                [r[key] for r in results_list],
                axis=0
            )
        
        for exp_name, results, c, m in itr:
            if(mode == "absolute"): 
                scores = results[key]
            elif(mode == "relative"): 
                scores = [
                    (np.array(r) - m) * 100 / m 
                    for r, m in zip(results[key], means)
                ]
            
            xs = list(range(1, len(scores) + 1))
            
            ax.set(
                xlabel="Num. Runs",
                **layouts[key]
            )
            ax.plot(xs, scores, label=exp_name, color=c, marker=m)
            ax.legend()
    
    return fig, axes


def mpl_mzml(mzml, 
             draw_minm=0.0, 
             colour_minm=None, 
             show_precursors=False):
    
    fig, ax = plt.subplots(1, 1)
    
    pp = PlotPoints.from_mzml(mzml)
    pp.mpl_add_ms1s(ax, draw_minm=draw_minm, colour_minm=colour_minm)
    ax.set( 
        xlabel="RT (Seconds)", 
        ylabel="m/z"
    )
    
    return fig, ax


def mpl_fragmentation_counts(evals, 
                             min_intensity=0.0, 
                             key="times_covered_summary", 
                             fcs=None):
    
    ylabel = "Count"
    if(key == "times_covered_summary"):
        title = "Times Peaks Covered"
        xlabel = "Times Covered"
    elif(key == "times_fragmented_summary"):
        title = "Times Peaks Fragmented"
        xlabel = "Times Fragmented"
    else:
        raise ValueError(f"Key {key} not recognised")
        
    if(fcs is None):
        fcs = itertools.repeat("skyblue")
    
    fig, axes = plt.subplots(1, len(evals), sharey=True)
    try: 
        axes[0]
    except:
        axes = [axes]
    
    for eva, ax, fc in zip(evals, axes, fcs):
        report = eva.evaluation_report(min_intensity=min_intensity)
        counter = report[key]
    
        fragmentations = [k for k, _ in counter.items()]
        counts = [v for _, v in counter.items()]

        ax.bar(
            x=fragmentations, 
            height=counts,
            align="center",
            width=1.0,
            fc=fc, 
            ec="black"
        )
    
        ax.set(title=title, xlabel=xlabel)
    axes[0].set(ylabel=ylabel)
    
    return fig, axes

def mpl_fragmentation_events(exp_name, 
                             mzmls, 
                             colour_minm=None):
    
    fig, axes = plt.subplots(len(mzmls), 1)
    
    for i, (mzml, ax) in enumerate(zip(mzmls, axes)):
        pp = PlotPoints.from_mzml(mzml)
        pp.mpl_add_ms2s(ax, colour_minm=colour_minm)
        ax.set(
            title=f"{exp_name} Run {i + 1} Fragmentation Events", 
            ylabel="m/z"
        )
    
    axes[-1].set(xlabel="RT (Seconds)")
    fig.set_size_inches(20, len(mzmls) * 4)
    
    return fig, axes


def mpl_fragmented_boxes_raw(exp_name, boxes):
    fig, ax = plt.subplots(1, 1)
    
    for b in boxes:
        b.mpl_add_to_plot(ax)
    ax.set(title=f"{exp_name} Picked Boxes", xlabel="RT (Seconds)", ylabel="m/z")
    ax.autoscale_view()
    
    return fig, ax
    

def mpl_fragmented_boxes(exp_name, eva, mode="max", min_intensity=0.0):
    """
        Only works with data read from .mzml, not env, for now
        Mode parameter unused for now, but could be used later
    """
    partition = PlotBox.from_evaluator(eva, min_intensity=min_intensity)
    boxes = partition["fragmented"] + partition["unfragmented"]
    return mpl_fragmented_boxes_raw(exp_name, boxes)


def plotly_results_plot(experiment_names, evals, min_intensity=0.0, suptitle=None):
    num_chems = len(evals[0].chems)
    results = [eva.evaluation_report(min_intensity=min_intensity) for eva in evals]
    repeat = max(r["num_runs"] for r in results)
    colours = itertools.cycle(DEFAULT_PLOTLY_COLORS)

    coverage_template = "<br>".join([
        "Iteration: %{x}",
        "Chems Fragmented: %{customdata} / " + str(num_chems),
        "Proportion: %{y} / 1.0"
    ])
    
    intensity_template = "<br>".join([
        "Iteration: %{x}",
        "Intensity Proportion: %{y} / 1.0"
    ]) 
    
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            "Multi-Sample Cumulative Coverage", 
            "Multi-Sample Cumulative Intensity Proportion"
        )
    )
    
    for exp_name, r, c in zip(experiment_names, results, colours):
        coverages = go.Scattergl(
            x=[i+1 for i, _ in enumerate(r["cumulative_coverage_proportion"])],
            y=r["cumulative_coverage_proportion"],
            marker={"color":c},
            line={"color":c},
            opacity=1,
            name=exp_name,
            customdata=r["sum_cumulative_coverage"],
            hovertemplate=coverage_template,
            legendgroup=exp_name
        )
        fig.add_trace(coverages, row=1, col=1)
        
        intensities = go.Scattergl(
            x=[i+1 for i, _ in enumerate(r["cumulative_intensity_proportion"])],
            y=r["cumulative_intensity_proportion"],
            marker={"color":c},
            line={"color":c},
            opacity=1,
            name=exp_name,
            hovertemplate=intensity_template,
            legendgroup=exp_name,
            showlegend=False
        )
        fig.add_trace(intensities, row=1, col=2)
        
    fig.update_xaxes(title_text="Num. Runs", range=[0, repeat + 1], row=1, col=1)
    fig.update_xaxes(title_text="Num. Runs", range=[0, repeat + 1], row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Coverage Proportion", range=[0.0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Intensity Proportion", range=[0.0, 1.1], row=1, col=2)
    
    fig.update_layout(
        template = "plotly_white",
    )
    
    fig.show()


def plotly_mzml(mzml, draw_minm=0.0, colour_minm=None, show_precursors=False):
    fig = go.Figure()
    
    mzml = path_or_mzml(mzml)
    pp = PlotPoints.from_mzml(mzml)
    fig.add_trace(
        pp.plotly_ms1s(
            draw_minm=draw_minm, 
            colour_minm=colour_minm, 
            show_precursors=show_precursors
        )
    )
    
    fig.update_layout(
        template="plotly_white",
        title = os.path.basename(mzml.file_name),
        xaxis_title="Retention Time",
        yaxis_title="m/z"
    )

    fig.show()


def plotly_timing_hist(processing_times, title, binsize=None):
    fig = make_subplots(rows=1, cols=len(processing_times))
    
    for i, run_times in enumerate(processing_times):
        fig.add_trace(
            go.Histogram(
                x=run_times,
                xbins={
                    "size": binsize
                },
                name=f"Injection {i}",
            ),
            row=1,
            col=i+1
        )

    fig.update_layout(
        template="plotly_white",
        title_text=f"{title} Scan Timings",
        xaxis_title_text="Processing Time (secs)",
        yaxis_title_text="Count",
    )

    fig.show()

 
def plotly_fragmentation_events(exp_name, mzmls, colour_minm=None):
    fig = make_subplots(rows=len(mzmls), cols=1, shared_xaxes="all", shared_yaxes="all")
    
    for i, mzml in enumerate(mzmls):
        mzml = path_or_mzml(mzml)
        pp = PlotPoints.from_mzml(mzml)
        data = pp.plotly_ms2s(colour_minm=colour_minm)
        data.name = os.path.basename(mzml.file_name)
        fig.add_trace(data, row=i+1, col=1)
        fig.update_yaxes(title_text="mz", row=i+1, col=1)
        fig.update_xaxes(title_text="Retention Time (seconds)", row=i+1, col=1)
    
    fig.update_layout(
        template = "plotly_white",
        title = f"{exp_name} Fragmentation Events",
        height = 200 * len(mzmls)
    )
    
    fig.show()


def seaborn_hist(data, xlabel, binsize=None):
    fig, axes = plt.subplots(1, len(data), figsize=(15, 5), sharey=True)
    
    try: axes[0]
    except: axes = [axes]
    
    for i, ts in enumerate(data):
        sns.histplot(ts, ax=axes[i], label=f"iter {i}", binwidth=binsize)
        axes[i].set(
            xlabel = xlabel,
            title = f"Iter {i}"
        )
    
    return fig, axes


def seaborn_mzml_timing_hist(mzmls, binsize=None, mode="combined"):
    if(mode == "combined"):
        timings = [get_scan_times_combined(mzmls)]
    else:
        timings = [get_scan_times(mzml) for mzml in mzmls]
        
    fig, axes = plt.subplots(len(timings), len(timings[0]))
    
    try: axes[0]
    except: axes = [axes]
    
    try: axes[0][0]
    except: axes = [axes]
    
    for mzml, times, ax_ls in zip(mzmls, timings, axes):
        for i, ax in enumerate(ax_ls):
            sns.histplot(times[i+1], ax=ax, label=f"MS{i+1}", binwidth=binsize)
            ax.set(
                xlabel = "Scan Duration",
                title = f"MS{i+1}"
            )
            
    for ax_ls in axes:
        ax_ls[0].set(ylabel="Count")
        for ax in ax_ls[1:]:
            ax.set(ylabel="")
            
    return fig, axes

def seaborn_timing_hist(processing_times, binsize=None):
    return seaborn_hist(
        processing_times, xlabel="Processing time (secs)", binsize=binsize
    )
    
    
def seaborn_uncovered_area_hist(eva, 
                                box_likes,
                                min_intensity=0.0, 
                                cumulative=False, 
                                binsize=None):
    
    boxes = [[b.to_box(0, 0) for b in ls] for ls in box_likes]    
    geom = BoxGrid()
    partition = eva.partition_chems(min_intensity=min_intensity, aggregate="max")
    
    figs = []
    labels = ["Fragmented", "Unfragmented"]
    for name in labels:
        non_overlaps = []
        for ls in boxes:
            if(not cumulative): geom.clear()
            geom.register_boxes(ls)
            non_overlaps.append(
                [geom.non_overlap(row[0]) for row in partition[name.lower()]]
            )
            
        fig, axes = seaborn_hist(non_overlaps, "Uncovered Area", binsize=binsize)    
        figs.append(
            (fig, axes, name)
        )
        
    return figs
        

class BoxViewer():
    def _check_length(self):
        max_len = max(
            len(self.mzmls),
            max(len(boxset) for boxset in self.boxes)
        )
        
        for boxset in self.boxes:
            for _ in range(len(boxset), max_len):
                boxset.append([])
                
        for _ in range(len(self.mzmls), max_len):
            self.mzmls.append(None)
            self.plot_points.append(PlotPoints([]))

    def __init__(self):
        self.headers = []
        self.boxes = []
        self.mzmls = []
        self.plot_points = []
        
    def summarise(self):
        self._check_length()
        print(f"BoxViewer of {len(self.mzmls)} runs")
        print(
            "\n".join(
                f"{h} of lengths: {[len(bs) for bs in row]}" 
                for h, row in zip(self.headers, self.boxes)
            )
        )
        
    def set_mzmls(self, mzmls):
        self.mzmls = [path_or_mzml(mzml) for mzml in mzmls]
        self.plot_points = [PlotPoints.from_mzml(mzml) for mzml in self.mzmls]
    
    def add_evaluator_boxes(self, evas, name="fragmented", min_intensity=0.0):
        new_boxes = []
        for eva in evas:
            new_boxes.append(PlotBox.from_evaluator(eva)[name])
        self.boxes.append(new_boxes)
        self.headers.append(f"{name} picked boxes")
    
    def add_roi_boxes(self, roi_lists, min_rt_width=0.0, min_mz_width=0.0):
        new_boxes = []
        for roi_ls in roi_lists:
            box_ls = []
            for roi in roi_ls:
                b = PlotBox.from_roi(roi, min_rt_width=min_rt_width, min_mz_width=min_mz_width)
                box_ls.append(b)
            new_boxes.append(box_ls)
        self.boxes.append(new_boxes)
        self.headers.append("rois")
        
    def add_geom_boxes(self, geoms):
        new_boxes = []
        for geom in geoms:
            new_boxes.append(PlotBox.from_grid(geom))
        self.boxes.append(new_boxes)
        self.headers.append("geom boxes")
        
    def _set_plot_bounds(self, box_index, boxset_index=0, rt_buffer=None, mz_buffer=None):
        xbounds = [float("inf"), float("-inf")]
        ybounds = [float("inf"), float("-inf")]

        for i, _ in enumerate(self.mzmls):
            chosen_box = self.boxes[boxset_index][i][box_index]
            new_xbounds, new_ybounds = chosen_box.get_plot_bounds(rt_buffer, mz_buffer)
            
            xbounds = [
                min(xbounds[0], new_xbounds[0]),
                max(xbounds[1], new_xbounds[1])
            ]
            
            ybounds = [
                min(ybounds[0], new_ybounds[0]),
                max(ybounds[1], new_ybounds[1])
            ]
        
        return xbounds, ybounds
        
    def summarise_box(self, box_index, boxset_index=0, rt_buffer=None, mz_buffer=None):
        self._check_length()
        xbounds, ybounds = self._set_plot_bounds(
            box_index, 
            boxset_index=boxset_index, 
            rt_buffer=rt_buffer, 
            mz_buffer=mz_buffer
        )
        
        for i, mzml in enumerate(self.mzmls):
            pts = self.plot_points[i]
            
            active_ms1, active_ms2 = pts.get_points_in_bounds(
                    min_rt=xbounds[0], 
                    max_rt=xbounds[1],
                    min_mz=ybounds[0], 
                    max_mz=ybounds[1]
            )
            
            if(not mzml is None):
                print(os.path.basename(mzml.file_name))
            else:
                print("No mzml")
            
            print(
                "MS1 Points:\n" + "\n".join(
                    f"rt: {rt}, m/z: {mz}, intensity: {intensity}"
                    for rt, mz, intensity in pts.ms1_points[active_ms1]
                )
            )
            
            print(
                "MS2 Points:\n" + "\n".join(
                    f"rt: {rt}, m/z: {mz}, intensity: {intensity}"
                    for rt, mz, intensity in pts.ms2s[active_ms2]
                )
            )
            
            for h, boxset in zip(self.headers, self.boxes):
                print(h)
                for b in boxset[i]:
                    in_bounds = b.box_in_bounds(
                        min_rt=xbounds[0], 
                        max_rt=xbounds[1],
                        min_mz=ybounds[0], 
                        max_mz=ybounds[1]
                    )
                    
                    if(in_bounds):
                        print(b)
                    
            print()
            print()

    def mpl_show_box(self, 
                     box_index, 
                     boxset_index=0,
                     rt_buffer=None, 
                     mz_buffer=None,
                     ms_level=1,
                     colour_minm=None,
                     abs_scaling=None):
        
        self._check_length()
        fig, axes = plt.subplots(len(self.mzmls), len(self.boxes))
        xbounds, ybounds = self._set_plot_bounds(
            box_index, 
            boxset_index=boxset_index, 
            rt_buffer=rt_buffer, 
            mz_buffer=mz_buffer
        )
        
        if(len(self.mzmls) == 1):
            axes = [axes]
        
        if(len(self.boxes) == 1):
            axes = [[ax] for ax in axes]
        
        for i, _ in enumerate(self.mzmls):
            pts = self.plot_points[i]
            
            for j, boxset in enumerate(self.boxes):
                
                if(ms_level == 1):
                    pts.mpl_add_ms1s(
                        axes[i][j], 
                        min_rt=xbounds[0], 
                        max_rt=xbounds[1],
                        min_mz=ybounds[0], 
                        max_mz=ybounds[1],
                        colour_minm=colour_minm,
                        abs_scaling=abs_scaling
                    )
                elif(ms_level == 2):
                    pts.mpl_add_ms2s(
                        axes[i][j], 
                        min_rt=xbounds[0], 
                        max_rt=xbounds[1],
                        min_mz=ybounds[0], 
                        max_mz=ybounds[1],
                        colour_minm=colour_minm,
                        abs_scaling=abs_scaling
                    )
                
                for b in boxset[i]:
                    in_bounds = b.box_in_bounds(
                        min_rt=xbounds[0], 
                        max_rt=xbounds[1],
                        min_mz=ybounds[0], 
                        max_mz=ybounds[1]
                    )
                    
                    if(in_bounds):
                        b.mpl_add_to_plot(axes[i][j])
                        
                axes[i][j].set_xlim(xbounds)
                axes[i][j].set_ylim(ybounds)
                axes[i][0].set(ylabel="m/z")
        
        for j, _ in enumerate(self.boxes):        
            axes[0][j].set(title=self.headers[j])
            axes[-1][j].set(xlabel="RT (Seconds)")
            
        fig.set_size_inches(20, len(self.mzmls) * 4)
        
        return fig, axes


def boxes2csv(fname, boxes, colours=None):
    """
        Write boxes to Batmass format
    """
    with open(fname, "w") as f:
        w = csv.writer(f)
        headers = ["rtLo", "rtHi", "mzLo", "mzHi", "intensity"]
        if(colours is None): 
            colours = itertools.repeat(None)
        else:
            headers.extend(["color", "opacity"])
        w.writerow(headers)
        
        for b, c in zip(boxes, colours):
            ls = b.serialise_info(minutes=True)
            if(not c is None):
                ls.extend([c.to_hexcode(), c.A])
            w.writerow(ls)

    
def csv2boxes(fname):
    """
        Read boxes from Batmass format
    """
    with open(fname, "r") as f:
        r = csv.DictReader(f)
        
        boxes, colours = [], []
        for row in r:
            boxes.append(
                GenericBox(
                    float(row["rtLo"]) * 60,
                    float(row["rtHi"]) * 60,
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