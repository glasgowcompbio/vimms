import math
import random
import bisect
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mass_spec_utils.data_import.mzml import MZMLFile

class RGBAColour():
    def __init__(self, R, G, B, A=1.0): self.R, self.G, self.B, self.A = R, G, B, A
    def __repr__(self): return f"RGBAColour(R={self.R}, G={self.G}, B={self.B}, A={self.A})"
    def __add__(self, other): return RGBAColour(self.R + other.R, self.G + other.G, self.B + other.B, self.A + other.A)
    def __mul__(self, scalar): return RGBAColour(self.R * scalar, self.G * scalar, self.B * scalar, self.A * scalar)
    def __rmul__(self, scalar): return self.__mul__(self.scalar)
    def squash(self): return (self.R / 255.0, self.G / 255.0, self.B / 255.0, self.A)

    def correct_bounds(self):
        self.R = max(0, min(255, self.R))
        self.G = max(0, min(255, self.G))
        self.B = max(0, min(255, self.B))
        self.A = max(0.0, min(1.0, self.A))
    
    def interpolate(self, others, weights=None):
        colours = [self] + others
        weights = [1 / len(colours) for _ in colours] if weights is None else weights
        new_c = sum((c * w for c, w in zip(colours, weights)), start=RGBAColour(0, 0, 0, 0.0))
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
            ax.add_patch(patches.Rectangle((box.pt1.x, box.pt1.y), box.pt2.x - box.pt1.x, box.pt2.y - box.pt1.y, linewidth=1, ec="black", fc=colour.squash()))

    def get_plot(self, boxes, key):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes, key)
        ax.set_xlim([min(b.pt1.x for b in boxes), max(b.pt2.x for b in boxes)])
        ax.set_ylim([min(b.pt1.y for b in boxes), max(b.pt2.y for b in boxes)])
        return plt
        
class FixedMap(ColourMap):
    '''Allows specifying a mapping from an enumeration of some property of interest to some list of colours.'''
    def __init__(self, mapping): 
        self.mapping = mapping
        
    def assign_colours(self, boxes, key):
        return ((b, self.mapping[min(key(b), len(self.mapping) - 1)]) for b in boxes)
        
    def unique_colours(self, boxes):
        return ((b, self.mapping[min(i, len(self.mapping) - 1)]) for i, b in enumerate(boxes))
        
class InterpolationMap(ColourMap):
    '''Allows assigning colours by interpolating between pairs of n colours specified across a value range of some property of interest.'''
    def __init__(self, colours):
        self.colours = list(colours)
        
    def assign_colours(self, boxes, key, minm=None, maxm=None):
        minm = min(key(b) for b in boxes) if minm is None else minm
        maxm = max(key(b) for b in boxes) if maxm is None else maxm
        intervals = [minm + i * (maxm - minm) / (len(self.colours) - 1) for i in range(len(self.colours))]
    
        def get_colour(box):
            if(math.isclose(minm, maxm)): return self.colours[0]
            i = next(i-1 for i, threshold in enumerate(intervals) if threshold >= key(box))
            weight = (key(box) - intervals[i]) / (intervals[i + 1] - intervals[i])
            return self.colours[i].interpolate([self.colours[i+1]], weights=[1-weight, weight])
        
        return ((b, get_colour(b)) for b in boxes)
        
class AutoColourMap(ColourMap):
    
    def __init__(self, colour_picker, reuse_colours=False):
        self.colour_picker = colour_picker
        self.reuse_colours = reuse_colours
        
    @staticmethod
    def random_colours(boxes):
        return ((b, RGBAColour(*(random.uniform(0, 255) for _ in range(3)))) for b in boxes)

    def assign_colours(self, boxes):
        #if there's no path from one box to another when we build a graph of their overlaps, we can re-use colours
        pairs = [(top, set()) for b in boxes for top in b.parents]
        top_level = OrderedDict(pairs) #need uniqueness and maintain ordering
        if(self.reuse_colours):
            for b in boxes:
                for top in b.parents: top_level[top].add(b) #note same set references in pairs and top_level
                
            components, indices = [], [-1 for _ in pairs]
            for i, (parent, children) in enumerate(top_level.items()):
                if(indices[i] == -1):
                    indices[i] = len(components)
                    components.append(OrderedDict([(parent, None)]))
                update = [(j, k) for j, (k, v) in enumerate(pairs[i:]) if children & v]
                for (j, k) in update:
                    indices[j] = indices[i]
                    components[indices[i]][k] = None
        else:
            components = [top_level]
                    
        top_level_colours = {top : colour for cs in components for top, colour in self.colour_picker(cs.keys())}
        def interpolate_lower(box): return top_level_colours[box.parents[0]].interpolate([top_level_colours[b] for b in box.parents[1:]])
        return ((b, interpolate_lower(b)) for b in boxes)
        
    def add_to_subplot(self, ax, boxes):
        for box, colour in self.assign_colours(boxes):
            ax.add_patch(patches.Rectangle((box.pt1.x, box.pt1.y), box.pt2.x - box.pt1.x, box.pt2.y - box.pt1.y, linewidth=1, ec="black", fc=colour.squash()))    
        
    def get_plot(self, boxes):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes)
        ax.set_xlim([min(b.pt1.x for b in boxes), max(b.pt2.x for b in boxes)])
        ax.set_ylim([min(b.pt1.y for b in boxes), max(b.pt2.y for b in boxes)])
        return plt

def path_or_mzml(mzml):    
    try:
        path = Path(mzml)
        mzml = MZMLFile(path)
    except:
        if(not type(mzml) == MZMLFile):
            raise NotImplementedError("Didn't recognise the MZMLFile!")
    return mzml
        
class PlotPoints():
    def __init__(self, ms1_points, ms2s=None, markers={}):
        self.ms1_points, self.ms2s = ms1_points, ms2s
        self.markers = markers
        self.active = np.ones_like(ms1_points)
        
    @staticmethod
    def from_mzml(mzml):
        mzml = path_or_mzml(mzml)
        scan_dict = {1: [], 2: []}
        for s in mzml.scans:
            if(s.ms_level in scan_dict): scan_dict[s.ms_level].append(s)
        scan_dict[1] = sorted(scan_dict[1], key=lambda s: s.rt_in_seconds)
        ms1_points = np.array([[s.rt_in_seconds, mz, intensity] for s in scan_dict[1] for mz, intensity in s.peaks])
        return PlotPoints(ms1_points, [(s.rt_in_seconds, s.precursor_mz) for s in scan_dict[2]])
        
    def get_points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        select_rt = ((min_rt is None) | (self.ms1_points[:, 0] >= min_rt)) & ((max_rt is None) | (self.ms1_points[:, 0] <= max_rt))  
        select_mz = ((min_mz is None) | (self.ms1_points[:, 1] >= min_mz)) & ((max_mz is None) | (self.ms1_points[:, 1] <= max_mz))
        return (select_rt & select_mz)
    
    def points_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        self.active = self.get_points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        
    def mark_precursors(self, max_error=10):
        markers = ["o" for _ in enumerate(self.ms1_points)]
        ms1_min, ms1_max = 0, 0
        for rt, precursor_mz in self.ms2s:
            ms1_max = bisect.bisect_right(self.ms1_points[:, 0], rt)
            precursor_time = self.ms1_points[ms1_max - 1, 0]
            ms1_min = bisect.bisect_left(self.ms1_points[:, 0], precursor_time)
            if(ms1_max - ms1_min > 0):
                i = np.argmin(np.abs(self.ms1_points[ms1_min:ms1_max, 1] - precursor_mz)) + ms1_min
                if(1e6 * np.abs(self.ms1_points[i, 1] - precursor_mz) / precursor_mz <= max_error):
                    markers[i] = "x"
        markers = np.array(markers)[self.active]
        marker_dict = {m : [] for m in set(markers)}
        for i, m in enumerate(markers):
            marker_dict[m].append(i)
        self.markers = marker_dict
        
    def plot_points(self, ax, min_rt=None, max_rt=None, min_mz=None, max_mz=None, abs_scaling=False):
        self.points_in_bounds(min_rt=min_rt, max_rt=max_rt, min_mz=min_mz, max_mz=max_mz)
        rts, mzs, intensities = self.ms1_points[self.active, 0], self.ms1_points[self.active, 1], self.ms1_points[self.active, 2]

        cmap = InterpolationMap([RGBAColour(238, 238, 238), ColourMap.YELLOW, ColourMap.RED, ColourMap.PURE_BLUE])        
        if(abs_scaling):
            minm, maxm = math.log(np.min(self.ms1_points[:, 2])), math.log(np.max(self.ms1_points[:, 2]))
            colours = np.array(list(cmap.assign_colours(intensities, lambda x: math.log(x), minm=minm, maxm=maxm)))
        else:
            colours = np.array(list(cmap.assign_colours(intensities, lambda x: math.log(x))))
        
        self.mark_precursors()
        for marker, idxes in self.markers.items():
            ax.scatter(rts[idxes], mzs[idxes], color=[colour.squash() for _, colour in colours[idxes]], marker=marker) 

class PlotBox():

    RT_TOLERANCE, MZ_TOLERANCE = 0.4, 0.4

    def __init__(self, min_rt, max_rt, min_mz, max_mz, intensity):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.min_mz, self.max_mz = min_mz, max_mz
        self.intensity = intensity
        
    @staticmethod
    def from_roi_aligner(aligner, ps=None):
        if(ps is None): pses = aligner.peaksets
        else: pses = [ps]
        
        plot_boxes = []
        for ps in pses:
            ps_boxes = []
            for box in aligner.peaksets2boxes[ps]:
                for p in ps.peaks:
                    rt_range, mz_range = box.rt_range_in_seconds, box.mz_range
                    pb = PlotBox(rt_range[0], rt_range[1], mz_range[0], mz_range[1], p.intensity)
                    ps_boxes.append(pb)
            plot_boxes.append(ps_boxes)
        
        return plot_boxes
        
    def __repr__(self): return f"PlotBox(min_mz={self.min_mz}, max_mz={self.max_mz}, min_rt={self.min_rt}, max_rt={self.max_rt}, apex_intensity={self.intensity})"
        
    def box_in_bounds(self, min_rt=None, max_rt=None, min_mz=None, max_mz=None):
        return (
                (min_rt is None or box.min_rt >= min_rt) and (max_rt is None or box.min_rt <= max_rt)
                or (min_rt is None or box.max_rt >= min_rt) and (max_rt is None or box.max_rt <= max_rt)
            and
            (
                (min_mz is None or box.min_mz >= min_mz) and (max_mz is None or box.min_mz <= box.max_mz)
                or (min_mz is None or box.max_mz >= min_mz) and (max_mz is None or box.max_mz <= box.max_mz)
            )
        )
        
    def add_to_plot(self, ax):
        x1, y1 = self.min_rt, self.min_mz
        xlen, ylen = (self.max_rt - self.min_rt), (self.max_mz - self.min_mz)
        ax.add_patch(patches.Rectangle((x1, y1), xlen, ylen, linewidth=1, ec="black", fc=[0, 0, 0, 0]))
        
    def get_plot_bounds(self):
        rt_buffer = (self.max_rt - self.min_rt) * self.RT_TOLERANCE
        mz_buffer = (self.max_mz - self.min_mz) * self.MZ_TOLERANCE
        xbounds = [self.min_rt - rt_buffer, self.max_rt + rt_buffer]
        ybounds = [self.min_mz - mz_buffer, self.max_mz + mz_buffer]
        return xbounds, ybounds

    def plot_box(self, ax, mzml, abs_scaling=False):
        xbounds, ybounds = self.get_plot_bounds()
        pts = PlotPoints.from_mzml(mzml)
        pts.plot_points(ax, xbounds[0], xbounds[1], ybounds[0], ybounds[1], abs_scaling=abs_scaling)
        self.add_to_plot(ax)
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        ax.set(xlabel="RT (Seconds)", ylabel="m/z")