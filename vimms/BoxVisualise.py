import os
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

from vimms.Common import path_or_mzml
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
                    # note same set references in pairs and top_level
                    top_level[top].add(b)

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
    hovertemplate = """
    Retention Time: %{x}<br>
    m/z %{y}<br>
    Intensity: %{customdata}
    """

    def __init__(self, ms1_points, ms2s=None, markers={}):
        self.ms1_points, self.ms2s = ms1_points, ms2s
        self.active_ms1 = np.ones((ms1_points.shape[0]))
        self.active_ms2 = np.ones((ms2s.shape[0]))
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
            max_mz=max_mz) if len(self.ms1_points) > 0 else np.array([[]])
        
        active_ms2 = self.bound_points(
            self.ms2s, 
            min_rt=min_rt, 
            max_rt=max_rt,
            min_mz=min_mz, 
            max_mz=max_mz) if len(self.ms2s) > 0 else np.array([[]])
        
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
        
    def colour_by_intensity(self, intensities, abs_scaling):
        cmap = InterpolationMap([
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED,
            ColourMap.PURE_BLUE
        ])
        
        if (abs_scaling):
            minm = math.log(np.min(self.ms1_points[:, 2]))
            maxm = math.log(np.max(self.ms1_points[:, 2]))
            colours = np.array(list(
                cmap.assign_colours(intensities, lambda x: math.log(x), minm=minm, maxm=maxm)
            ))
        else:
            colours = np.array(list(
                cmap.assign_colours(intensities, lambda x: math.log(x))
            ))
        
        return colours

    def mpl_add_ms1s(self, ax, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None,
                    min_intensity=0.0,
                    abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        self.mark_precursors()
        
        active_ms1 = self.ms1_points[self.active_ms1, :]
        sort_idx = np.argsort(active_ms1[:, 2])
        srted = active_ms1[sort_idx]
        filter_idx = srted[:, 2] >= min_intensity
        rts, mzs, intensities = srted[filter_idx].T
        markers = np.array(self.markers)[self.active_ms1][sort_idx][filter_idx]
        markers = ["o" if m == "0" else m for m in markers]
        
        colours = self.colour_by_intensity(intensities, abs_scaling=abs_scaling)
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
                    abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        active_ms2 = self.ms2s[self.active_ms2, :]
        rts, mzs, intensities = active_ms2[np.argsort(active_ms2[:, 2])].T
        colours = self.colour_by_intensity(intensities, abs_scaling=abs_scaling)
        ax.scatter(rts, mzs, color=[colour.squash() for _, colour in colours], marker="x")

    def plotly_ms1s(self, 
                    min_rt=None, max_rt=None, 
                    min_mz=None, max_mz=None, 
                    min_intensity=0.0,
                    show_precursors=False,
                    abs_scaling=False):
                    
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        self.mark_precursors()
        
        active_ms1 = self.ms1_points[self.active_ms1, :]
        sort_idx = np.argsort(active_ms1[:, 2])
        srted = active_ms1[sort_idx]
        filter_idx = srted[:, 2] >= min_intensity
        rts, mzs, intensities = active_ms1[filter_idx].T
        
        lit = [math.log(it) for it in intensities]
        colourscale = [
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED, 
            ColourMap.PURE_BLUE
        ]
        
        if(show_precursors):
            markers = np.array(self.markers)[self.ms1_points[:, 2] >= min_intensity][sort_idx]
        else:
            markers = "0"
        
        return go.Scattergl(
                x=rts, 
                y=mzs,
                mode="markers",
                marker=dict(
                    symbol=markers,
                    color=lit,
                    colorscale=[c.to_plotly()[0] for c in colourscale],
                    cmin=min(np.log(active[:, 2])) if abs_scaling else None,
                    cmax=max(np.log(active[:, 2])) if abs_scaling else None
                ),
                hovertemplate=self.hovertemplate,
                customdata=intensities
        )

    def plotly_ms2s(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None, abs_scaling=False):   
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        active_ms2 = self.ms2s[self.active_ms2, :]
        rts, mzs, intensities = active_ms2[np.argsort(active_ms2[:, 2])].T
        lit = [math.log(it) for it in intensities]
        colourscale = [
            RGBAColour(238, 238, 238), 
            ColourMap.YELLOW, 
            ColourMap.RED, 
            ColourMap.PURE_BLUE
        ]
        
        return go.Scattergl(
                x=rts, 
                y=mzs,
                mode="markers",
                marker=dict(
                    symbol="x",
                    color=lit,
                    colorscale=[c.to_plotly()[0] for c in colourscale],
                    cmin=min(lit),
                    cmax=max(lit)
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
                     rt_buffer=None, 
                     mz_buffer=None):
        xbounds, ybounds = self.get_plot_bounds(rt_buffer=rt_buffer,
                                                mz_buffer=mz_buffer)
        pts = PlotPoints.from_mzml(mzml)
        pts.mpl_add_ms1s(ax, min_rt=xbounds[0], max_rt=xbounds[1],
                      min_mz=ybounds[0], max_mz=ybounds[1],
                      abs_scaling=abs_scaling)
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
            roi_builder = env.controller.roi_builder
            live_roi, dead_roi, junk_roi = (
                roi_builder.live_roi, roi_builder.dead_roi, roi_builder.junk_roi
            )
            self.rois = live_roi + dead_roi + junk_roi
        except AttributeError:
            pass
        
        try:
            self.bm = env.controller.grid
        except AttributeError:
            pass
        
        
def mpl_results_plot(experiment_names, evals, min_intensity=0.0, suptitle=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for exp_name, eva in zip(experiment_names, evals):
        results = eva.evaluation_report(min_intensity=min_intensity)
        coverages = results["cumulative_coverage_proportion"]
        intensity_proportions = results["cumulative_intensity_proportion"]
        xs = list(range(1, len(coverages) + 1))

        ax1.set(
            xlabel="Num. Runs", 
            ylabel="Cumulative Coverage Proportion", 
            title="Multi-Sample Cumulative Coverage"
        )
        ax1.plot(xs, coverages, label=exp_name)
        ax1.legend()

        ax2.set(
            xlabel="Num. Runs", 
            ylabel="Cumulative Intensity Proportion", 
            title="Multi-Sample Cumulative Intensity Proportion"
        )
        ax2.plot(xs, intensity_proportions, label=exp_name)
        ax2.legend()
    
    fig.set_size_inches(18.5, 10.5)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=18)
    
    plt.show()


def mpl_mzml(mzml, min_intensity=0.0, show_precursors=False):
    fig, ax = plt.subplots(1, 1)
    
    pp = PlotPoints.from_mzml(mzml)
    pp.mpl_add_ms1s(ax)
    ax.set( 
        xlabel="RT (Seconds)", 
        ylabel="m/z"
    )
    fig.set_size_inches(18.5, 10.5)
    plt.plot()
    

def mpl_fragmentation_events(exp_name, mzmls):
    fig, axes = plt.subplots(len(mzmls), 1)
    
    for i, (mzml, ax) in enumerate(zip(mzmls, axes)):
        pp = PlotPoints.from_mzml(mzml)
        pp.mpl_add_ms2s(ax)
        ax.set(
            title=f"{exp_name} Run {i + 1} Fragmentation Events", 
            ylabel="m/z"
        )
    
    axes[-1].set(xlabel="RT (Seconds)")
    fig.set_size_inches(20, len(mzmls) * 4)
    plt.plot()


def mpl_fragmented_boxes_raw(exp_name, boxes):
    fig, ax = plt.subplots(1, 1)
    for b in boxes:
        b.mpl_add_to_plot(ax)
    ax.set(title=f"{exp_name} Picked Boxes", xlabel="RT (Seconds)", ylabel="m/z")
    fig.set_size_inches(20, 10)
    plt.plot()
    

def mpl_fragmented_boxes(exp_name, eva, mode="max", min_intensity=0.0):
    """
        Only works with data read from .mzml, not env, for now
        Mode parameter unused for now, but could be used later
    """
    partition = PlotBox.from_evaluator(eva, min_intensity=min_intensity)
    boxes = partition["fragmented"] + partition["unfragmented"]
    mpl_fragmented_boxes_raw(exp_name, boxes)


def plotly_results_plot(experiment_names, evals, min_intensity=0.0, suptitle=None):
    num_chems = len(evals[0].chems)

    coverage_template = "<br>".join([
        "Iteration: %{x}",
        "Chems Fragmented: %{customdata} / " + str(num_chems),
        "Proportion: %{y} / 1.0"
    ])
    
    intensity_template = "<br>".join([
        "Iteration: %{x}",
        "Proportion: %{y} / 1.0"
    ]) 

    results = [eva.evaluation_report(min_intensity=min_intensity) for eva in evals]
    
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=(
            "Multi-Sample Cumulative Coverage", 
            "Multi-Sample Cumulative Intensity Proportion"
        )
    )
    
    for exp_name, r, c in zip(experiment_names, results, DEFAULT_PLOTLY_COLORS):
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
        
    fig.update_xaxes(title_text="Num. Runs", range=[0, 7], row=1, col=1)
    fig.update_xaxes(title_text="Num. Runs", range=[0, 7], row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Coverage Proportion", range=[0.0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Intensity Proportion", range=[0.0, 1.1], row=1, col=2)
    
    fig.update_layout(
        template = "plotly_white",
    )
    
    fig.show()


def plotly_mzml(mzml, min_intensity=0.0, show_precursors=False):
    fig = go.Figure()
    
    mzml = path_or_mzml(mzml)
    pp = PlotPoints.from_mzml(mzml)
    fig.add_trace(pp.plotly_ms1s(min_intensity=min_intensity, show_precursors=show_precursors))
    
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

 
def plotly_fragmentation_events(exp_name, mzmls):
    fig = make_subplots(rows=len(mzmls), cols=1, shared_xaxes="all", shared_yaxes="all")
    
    for i, mzml in enumerate(mzmls):
        mzml = path_or_mzml(mzml)
        pp = PlotPoints.from_mzml(mzml)
        data = pp.plotly_ms2s()
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


def seaborn_hist(data, title, xlabel, binsize=None):
    fig, ax = plt.subplots(1, len(data), figsize=(15, 5), sharey=True)
    plt.suptitle(title, fontsize=18)
    for i, ts in enumerate(data):
        sns.histplot(ts, ax=ax[i], label=f"iter {i}", binwidth=binsize)
        ax[i].set(
            xlabel = xlabel,
            title = f"Iter {i}"
        )


def seaborn_timing_hist(processing_times, title, binsize=None):
    seaborn_hist(processing_times, title, xlabel="Processing time (secs)", binsize=binsize)
    
    
def seaborn_uncovered_area_hist(eva, 
                                box_likes, 
                                title, 
                                min_intensity=0.0, 
                                cumulative=False, 
                                binsize=None):
    boxes = [[b.to_box(0, 0) for b in ls] for ls in box_likes]    
    geom = BoxGrid()
    partition = eva.partition_chems(min_intensity=min_intensity, aggregate="max")
    
    for name in ["fragmented", "unfragmented"]:
        non_overlaps = []
        for ls in boxes:
            if(not cumulative): geom.clear()
            geom.register_boxes(ls)
            non_overlaps.append(
                [geom.non_overlap(row[0]) for row in partition[name]]
            )
        seaborn_hist(non_overlaps, title + f" ({name})", "Uncovered Area", binsize=binsize)
        

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
        
    def describe(self):
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

    def mpl_show_box(self, 
                     box_index, 
                     boxset_index=0,
                     rt_buffer=None, 
                     mz_buffer=None,
                     abs_scaling=None,
                     suptitle=None):
        
        self._check_length()
        xbounds = [float("inf"), float("-inf")]
        ybounds = [float("inf"), float("-inf")]
        fig, axes = plt.subplots(len(self.mzmls), len(self.boxes))

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
        
        for i, _ in enumerate(self.mzmls):
            pts = self.plot_points[i]
            
            for j, boxset in enumerate(self.boxes):
                pts.mpl_add_ms1s(
                    axes[i][j], 
                    min_rt=xbounds[0], 
                    max_rt=xbounds[1],
                    min_mz=ybounds[0], 
                    max_mz=ybounds[1],
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
            
        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=18)
        fig.set_size_inches(20, len(self.mzmls) * 4)
        
        plt.show()


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