"""
Note: this module is still under development and might change significantly.
"""
import copy
import itertools
import math
import random
import itertools
from abc import abstractmethod, ABCMeta
from collections import defaultdict, namedtuple
from decimal import Decimal
from functools import reduce
from operator import attrgetter

import intervaltree
import numpy as np
from mass_spec_utils.data_import.mzmine import PickedBox

class Point():
    def round(self, ndigits=8):
        self.x, self.y = round(self.x, ndigits), round(self.y, ndigits)

    def __init__(self, x, y): 
        self.x, self.y = float(x), float(y)
        self.round()

    def __eq__(self, other_point): 
        return (
            type(self) == type(other_point) 
            and math.isclose(self.x, other_point.x) 
            and math.isclose(self.y, other_point.y)
        )
        
    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __repr__(self): 
        return "Point({}, {})".format(self.x, self.y)


class Interval():
    errmsg = "Interval has to be flat on one of the dimensions!"

    def __init__(self, x1, x2, y1, y2):
        assert math.isclose(x1, x2) or math.isclose(y1, y2), self.errmsg
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
    
    def __eq__(self, other_inv):
        return (
            type(self) == type(other_inv) 
            and self.pt1 == other_inv.pt1 
            and self.pt2 == other_inv.pt2
        )
        
    def __hash__(self):
        return (self.pt1, self.pt2).__hash__()

    def __repr__(self):
        return "Interval({}, {})".format(self.pt1, self.pt2)        
    
    def is_vertical(self):
        return math.isclose(self.pt1.x, self.pt2.x)
        
    def is_horizontal(self):
        return math.isclose(self.pt1.y, self.pt2.y)


class Box():
    def __init__(self, x1, x2, y1, y2, 
                    parents=None, 
                    min_xwidth=0, 
                    min_ywidth=0, 
                    intensity=0, 
                    id=None, 
                    roi=None):
                
        self.id = id
        self.roi = roi
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
        self.parents = [self] if parents is None else parents
        self.intensity = intensity

        if (self.pt2.x - self.pt1.x < min_xwidth):
            midpoint = self.pt1.x + (self.pt2.x - self.pt1.x) / 2
            self.pt1.x = midpoint - min_xwidth / 2 
            self.pt2.x = midpoint + min_xwidth / 2

        if (self.pt2.y - self.pt1.y < min_ywidth):
            midpoint = self.pt1.y + (self.pt2.y - self.pt1.y) / 2
            self.pt1.y = midpoint - min_ywidth / 2 
            self.pt2.y = midpoint + min_ywidth / 2
            
        self.pt1.round()
        self.pt2.round()

    def __repr__(self):
        return "Box({}, {})".format(self.pt1, self.pt2)

    def __eq__(self, other_box):
        return (
            type(self) == type(other_box) 
            and self.pt1 == other_box.pt1 
            and self.pt2 == other_box.pt2
        )

    def __hash__(self):
        return (self.pt1, self.pt2).__hash__()
        
    def serialise_info(self):
        return [self.min_rt, self.max_rt, self.min_mz, self.max_mz, self.intensity]

    def area(self):
        return (self.pt2.x - self.pt1.x) * (self.pt2.y - self.pt1.y)

    def copy(self, xshift=0, yshift=0):
        return type(self)(
                self.pt1.x + xshift, 
                self.pt2.x + xshift, 
                self.pt1.y + yshift, 
                self.pt2.y + yshift,
                parents=self.parents, 
                intensity=self.intensity, 
                id=self.id, 
                roi=self.roi
               )

    def shift(self, xshift=0, yshift=0):
        self.pt1.x += xshift
        self.pt2.x += xshift
        self.pt1.y += yshift
        self.pt2.y += yshift
        self.pt1.round()
        self.pt2.round()

    def num_overlaps(self):
        return len(self.parents)
        
    @classmethod
    def from_pickedbox(cls, pbox):
        min_rt, max_rt = pbox.rt_range_in_seconds
        min_mz, max_mz = pbox.mz_range
        return cls(min_rt, max_rt, min_mz, max_mz)

    def to_pickedbox(self, peak_id):
        rts = [self.pt1.x, self.pt2.x]
        mzs = [self.pt1.y, self.pt2.y]
        return PickedBox(peak_id, sum(mzs) / 2, sum(rts) / 2, mzs[0], mzs[1],
                         rts[0], rts[1])


class GenericBox(Box):
    """Makes no particular assumptions about bounding boxes."""

    def __repr__(self):
        return "Generic{}".format(super().__repr__())

    def contains_point(self, pt):
        return (
                self.pt1.x <= pt.x
                and self.pt1.y <= pt.y
                and self.pt2.x >= pt.x
                and self.pt2.y >= pt.y
        )
        
    def interval_contains(self, inv):
        if(inv.is_vertical()):
            return (
                self.pt1.x <= inv.pt1.x
                and self.pt2.x >= inv.pt1.x
                and self.pt1.y >= inv.pt1.y
                and self.pt2.y <= inv.pt2.y
            )
        else:
            return (
                self.pt1.x >= inv.pt1.x
                and self.pt2.x <= inv.pt2.x
                and self.pt1.y <= inv.pt1.y
                and self.pt2.y >= inv.pt1.y
            )

    def overlaps_with_box(self, other_box):
        return (
            self.pt1.x < other_box.pt2.x 
            and self.pt2.x > other_box.pt1.x 
            and self.pt1.y < other_box.pt2.y 
            and self.pt2.y > other_box.pt1.y
        )

    def contains_box(self, other_box):
        return (
            self.pt1.x <= other_box.pt1.x
            and self.pt1.y <= other_box.pt1.y
            and self.pt2.x >= other_box.pt2.x
            and self.pt2.y >= other_box.pt2.y
        )
        
    def overlap_raw(self, other_box):
        if(not self.overlaps_with_box(other_box)): 
            return 0.0
        return (
            (min(self.pt2.x, other_box.pt2.x) - max(self.pt1.x, other_box.pt1.x))
            * (min(self.pt2.y, other_box.pt2.y) - max(self.pt1.y, other_box.pt1.y))
        )

    def overlap_2(self, other_box):
        if(not self.overlaps_with_box(other_box)): 
            return 0.0
        b = type(self)(
                max(self.pt1.x, other_box.pt1.x), 
                min(self.pt2.x, other_box.pt2.x), 
                max(self.pt1.y, other_box.pt1.y),
                min(self.pt2.y, other_box.pt2.y)
            )
        return b.area() / (self.area() + other_box.area() - b.area())

    def overlap_3(self, other_box):
        if(not self.overlaps_with_box(other_box)): 
            return 0.0
        b = type(self)(
                max(self.pt1.x, other_box.pt1.x), 
                min(self.pt2.x, other_box.pt2.x), 
                max(self.pt1.y, other_box.pt1.y),
                min(self.pt2.y, other_box.pt2.y)
            )
        return b.area() / self.area()

    def non_overlap_split(self, other_box):
        """Finds 1 to 4 boxes describing the polygon of area of this box
        not overlapped by other_box. If one box is found, crops this box to
        dimensions of that box, and returns None. Otherwise, returns list of
        2 to 4 boxes. Number of boxes found is equal to number of edges
        overlapping area does NOT share with this box."""
        if (not self.overlaps_with_box(other_box)):
            return None
        x1, x2, y1, y2 = self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y
        split_boxes = []
        if (other_box.pt1.x > self.pt1.x):
            x1 = other_box.pt1.x
            split_boxes.append(
                type(self)(self.pt1.x, x1, y1, y2, parents=self.parents, intensity=self.intensity)
            )
        if (other_box.pt2.x < self.pt2.x):
            x2 = other_box.pt2.x
            split_boxes.append(
                type(self)(x2, self.pt2.x, y1, y2, parents=self.parents, intensity=self.intensity)
            )
        if (other_box.pt1.y > self.pt1.y):
            y1 = other_box.pt1.y
            split_boxes.append(
                type(self)(x1, x2, self.pt1.y, y1, parents=self.parents, intensity=self.intensity)
            )
        if (other_box.pt2.y < self.pt2.y):
            y2 = other_box.pt2.y
            split_boxes.append(
                type(self)(x1, x2, y2, self.pt2.y, parents=self.parents, intensity=self.intensity)
            )
        return split_boxes

    def split_all(self, other_box):
        if (not self.overlaps_with_box(other_box)):
            return None, None, None
        both_parents = self.parents + other_box.parents
        both_box = type(self)(
                        max(self.pt1.x, other_box.pt1.x), 
                        min(self.pt2.x, other_box.pt2.x),
                        max(self.pt1.y, other_box.pt1.y), 
                        min(self.pt2.y, other_box.pt2.y), 
                        parents=both_parents,
                        intensity=max(self.intensity, other_box.intensity)
                   )
        b1_boxes = self.non_overlap_split(other_box)
        b2_boxes = other_box.non_overlap_split(self)
        return b1_boxes, b2_boxes, both_box

class Grid(metaclass=ABCMeta):
    '''
        Partitions a 2D space into a number of rectangles of fixed size, for faster lookup.
        If a query object and a saved object touch the same rectangle, then the saved object should be factored into the query.
    '''

    @staticmethod
    @abstractmethod
    def init_boxes(): pass

    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz,
                 mz_box_size):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.min_mz, self.max_mz = min_mz, max_mz
        self.rt_box_size, self.mz_box_size = rt_box_size, mz_box_size
        self.box_area = float(Decimal(rt_box_size) * Decimal(mz_box_size))

        self.rtboxes = range(0, int((self.max_rt - self.min_rt) /
                                    rt_box_size) + 1)
        self.mzboxes = range(0, int((self.max_mz - self.min_mz) /
                                    mz_box_size) + 1)
        self.boxes = self.init_boxes(self.rtboxes, self.mzboxes)

    def get_box_ranges(self, box):
        rt_box_range = (
            int((box.pt1.x - self.min_rt) / self.rt_box_size), 
            int((box.pt2.x - self.min_rt) / self.rt_box_size) + 1
        )
        mz_box_range = (
            int((box.pt1.y - self.min_mz) / self.mz_box_size), 
            int((box.pt2.y - self.min_mz) / self.mz_box_size) + 1
        )
        total_boxes = (
            (rt_box_range[1] - rt_box_range[0]) 
            * (mz_box_range[1] - mz_box_range[0])
        )
        return rt_box_range, mz_box_range, total_boxes

    @abstractmethod
    def register_box(self, box): pass

    def clear(self):
        self.__init__(
            self.min_rt, self.max_rt, self.rt_box_size, 
            self.min_mz, self.max_mz, self.mz_box_size
        )
        

class DictGrid(Grid):
    '''
        A sparse, lossless implementation of the grid.
    '''

    @staticmethod
    def init_boxes(rtboxes, mzboxes):
        return defaultdict(list)

    def approx_non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        active = sum(
                float(not self.boxes[(rt, mz)]) 
                for rt in range(*rt_box_range) 
                for mz in range(*mz_box_range)
        )
        return active / total_boxes

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for rt in range(*rt_box_range):
            for mz in range(*mz_box_range):
                self.boxes[(rt, mz)].append(box)


class ArrayGrid(Grid):
    '''
        A dense, lossy implementation of the grid.
    '''

    @staticmethod
    def init_boxes(rtboxes, mzboxes): 
        return np.array([[False for mz in mzboxes] for rt in rtboxes])

    def approx_non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        boxes = self.boxes[rt_box_range[0]:rt_box_range[1],
                mz_box_range[0]:mz_box_range[1]]
        return (total_boxes - np.sum(boxes)) / total_boxes

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        self.boxes[rt_box_range[0]:rt_box_range[1],
        mz_box_range[0]:mz_box_range[1]] = True


class LocatorGrid(Grid):
    '''
        A dense, lossless implementation of the grid.
    '''

    @staticmethod
    def init_boxes(rtboxes, mzboxes):
        arr = np.empty((max(rtboxes), max(mzboxes)), dtype=object)
        for i, row in enumerate(arr):
            for j, _ in enumerate(row):
                arr[i, j] = set()
        return arr

    def get_boxes(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        boxes = set()
        for row in self.boxes[rt_box_range[0]:rt_box_range[1],
                   mz_box_range[0]:mz_box_range[1]]:
            for s in row:
                boxes |= s
        return boxes

    def get_all_boxes(self):
        return reduce(lambda s1, s2: s1 | s2, (s for row in self.boxes for s in row))

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for s in row:
                s.add(box)
                
    def clear(self):
        for i, row in enumerate(self.boxes):
            for j, _ in enumerate(row):
                self.boxes[i, j] = set()

class LineSweeper():
    EndPoint = namedtuple("EndPoint", ["pos", "is_end", "box"])
    
    def __init__(self):
        self.all_boxes, self.active, self.was_active = [], [], []
        self.active_intervals = intervaltree.IntervalTree()
        self.previous_loc, self.current_loc = -1, 0
        self.active_pointer = 0
        self.removed = set()

    def get_all_boxes(self):
        return self.all_boxes
        
    def _add_active(self, new_ptr, current_loc):
        for b in itertools.islice(self.all_boxes, self.active_pointer, new_ptr):
            self.active.append(b)
            self.active_intervals.addi(b.pt1.y, b.pt2.y + 1E-12, b)
        self.active.sort(key=attrgetter("pt2.x"), reverse=True)
        self.active_pointer = new_ptr
    
    def _remove_active(self):
        b = self.active.pop()
        self.was_active.append(b)
        if(not b in self.removed):
            self.active_intervals.removei(b.pt1.y, b.pt2.y + 1E-12, b)
            self.removed.add(b)
    
    def set_active_boxes(self, current_loc):
        if(current_loc < self.current_loc):
            self.active_pointer = 0
            self.active = []
            self.active_intervals = intervaltree.IntervalTree()
            self.previous_loc = -1
            self.current_loc = 0
            self.removed = set()
            
        new_ptr = self.active_pointer
        while(new_ptr < len(self.all_boxes) and self.all_boxes[new_ptr].pt1.x <= current_loc):
            new_ptr += 1
        self._add_active(new_ptr, current_loc)
        
        self.was_active = []
        while(self.active != [] and self.active[-1].pt2.x <= current_loc):
            self._remove_active()
        self.previous_loc = self.current_loc
        self.current_loc = current_loc
        
    def point_in_box(self, pt):
        return self.active_interval.overlaps_point(pt.y)
        
    def point_in_which_boxes(self, pt):
        return [inv.data for inv in self.active_intervals.at(pt.y)]
        
    def interval_overlaps_which_boxes(self, inv):
        return [inv.data for inv in self.active_intervals.overlap(inv.pt1.y, inv.pt2.y)]
        
    def interval_covers_which_boxes(self, inv):
        return [inv.data for inv in self.active_intervals.envelop(inv.pt1.y, inv.pt2.y)]
    
    @classmethod
    def _to_endpoint(cls, boxes, coord):
        eps = []
        for b in boxes:
            eps.append(cls.EndPoint(getattr(b.pt1, coord), False, b))
            eps.append(cls.EndPoint(getattr(b.pt2, coord), True, b))
        eps.sort(key=lambda ep: ep[:2])
        return eps
       
    @staticmethod
    def _group_endpoints(endpts):
        return (
            list(v) for k, v in itertools.groupby(endpts, key=attrgetter("pos"))
        )
        
    @staticmethod
    def _unify_intervals(endpts):
        begin, count = -1, 0
        for pos, end, _ in endpts:
            if(end):
                count -= 1
                if(count == 0): yield (begin, pos)
            else:
                if(count == 0): begin = pos
                count += 1
   
    def split_all_boxes(self):
        new_id, new_boxes = 0, []
        prev_intervals = intervaltree.IntervalTree()
        
        x_ends = self._to_endpoint(self.all_boxes, "x")
        for x_group in self._group_endpoints(x_ends):
            for x_pos, x_end, x_box in x_group:
                if(x_end): self._remove_active()
                else: self._add_active(self.active_pointer + 1, x_pos)
        
            x_pos, _, __ = x_group[0]
            to_query = self._to_endpoint((x.box for x in x_group), "y")
            
            invs = self._unify_intervals(to_query)
            for begin, end in invs:
                for inv in prev_intervals.overlap(begin, end - 1E-12):
                    prev_x_pos, parents = inv.data
                    new_boxes.append(
                        GenericBox(
                            prev_x_pos,
                            x_pos,
                            max(begin, inv.begin),
                            min(end, inv.end),
                            parents = parents,
                            intensity = max(b.intensity for b in parents),
                            id = new_id
                        )
                    )
                    new_id += 1
                    prev_intervals.chop(begin, end)
            
                overlapped = self.interval_overlaps_which_boxes(
                    Interval(x_pos, x_pos, begin + 1E-12, end - 1E-12)
                )
                y_ends = self._to_endpoint(overlapped, "y")
                
                prev_y_pos, y_active = -1, []
                for y_group in self._group_endpoints(y_ends):
                    y_pos, _, __ = y_group[0]
                          
                    if(prev_y_pos >= end): break
                    if(y_active != [] and y_pos > begin):
                        prev_intervals.addi(
                            max(begin, prev_y_pos),
                            min(end, y_pos), 
                            (x_pos, copy.copy(y_active))
                        )
                            
                    for y_pos, y_end, y_box in y_group:
                        if(y_end): y_active.remove(y_box)
                        else: y_active.append(y_box)
                    
                    prev_y_pos = y_pos
                
        return new_boxes
        
    def register_boxes(self, boxes):
        self.all_boxes = sorted(itertools.chain(self.all_boxes, boxes), key=attrgetter("pt1.x"))

class BoxGeometry(metaclass=ABCMeta):
    '''
        Describes the interface for an abstract class which can do geometric operations on points, intervals and rectangles.
        Different subclasses use different data structures, and hence the choice of data structure matters for performance.
    '''
    
    def set_active_boxes(self, *args):
        pass
        
    @abstractmethod
    def get_all_boxes(self):
        pass

    @abstractmethod
    def point_in_box(self, pt):
        pass
        
    @abstractmethod
    def point_in_which_boxes(self, pt):
        pass
        
    @abstractmethod
    def interval_overlaps_which_boxes(self, inv):
        pass
        
    @abstractmethod
    def interval_covers_which_boxes(self, inv):
        pass
        
    @abstractmethod
    def non_overlap(self, box): 
        pass
    
    @abstractmethod
    def intensity_non_overlap(self, box): 
        pass
        
    @abstractmethod
    def register_boxes(self, boxes):
        pass
        
    @abstractmethod
    def clear(self):
        pass

class BoxApproximator(BoxGeometry):

    def get_all_boxes(self):
        raise NotImplementedError("Approximator methods don't implement this method!")

    def point_in_box(self, pt):
        raise NotImplementedError("Approximator methods don't implement this method!")
        
    def point_in_which_boxes(self, pt):
        raise NotImplementedError("Approximator methods don't implement this method!")
        
    def interval_overlaps_which_boxes(self, inv):
        raise NotImplementedError("Approximator methods don't implement this method!")
        
    def interval_covers_which_boxes(self, inv):
        raise NotImplementedError("Approximator methods don't implement this method!")

    def non_overlap(self, box): 
        self.grid.approx_non_overlap(box)
    
    def intensity_non_overlap(self, box):
        #TODO: approximators don't implement this yet
        raise NotImplementedError("Approximator methods don't implement this method yet!")
        self.grid.intensity_non_overlap(box)

    def register_boxes(self, boxes):
        for b in boxes:
            self.grid.register_box(b)
    
    def clear(self):
        self.grid.clear()
        
        
class BoxApproximatorDict(BoxApproximator):
    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size):
        self.grid = DictGrid(min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size)


class BoxApproximatorArray(BoxApproximator):
    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size):
        self.grid = ArrayGrid(min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size)
        
        
class BoxExact(BoxGeometry):
    @abstractmethod
    def _get_overlapping_boxes(self, box):
        pass

    @staticmethod
    def non_overlap_boxes(box, other_boxes):
        new_boxes = [box]
        for b in other_boxes:
            if(box.overlaps_with_box(b)): #quickly exits any box not overlapping new box
                updated_boxes = []
                for b2 in new_boxes:
                    #if your box is contained within a
                    #previous box area is 0 and box is not carried over
                    if(not b.contains_box(b2)):  
                        split_boxes = b2.non_overlap_split(b)
                        if(split_boxes is not None):
                            updated_boxes.extend(split_boxes)
                        else:
                            updated_boxes.append(b2)
                if(not updated_boxes):
                    return []
                new_boxes = updated_boxes
        return new_boxes

    @staticmethod
    def splitting_non_overlap(box, other_boxes):
        boxes = BoxExact.non_overlap_boxes(box, other_boxes)
        if(boxes == []): return 0.0
        else: return sum(b.area() for b in boxes) / box.area()
        
    def non_overlap(self, box):
        return float(self.splitting_non_overlap(box, self._get_overlapping_boxes(box)))
        
    @staticmethod
    def splitting_intensity_non_overlap(box, 
                                        other_boxes, 
                                        current_intensity, 
                                        scoring_params
                                       ):
        """Will give nonsense results if other_boxes overlap each other."""
        areas = [box.overlap_raw(b) for b in other_boxes]
        non_overlapped = max(0.0, 1.0 - sum(areas) / box.area())
        non_overlap = current_intensity ** non_overlapped
        refragment = sum(
            max(0.0, current_intensity - b.intensity) ** (area / box.area())
            for b, area in zip(other_boxes, areas)
        )
        return (
            non_overlap 
            + scoring_params["theta1"] * refragment
        )
        
    def intensity_non_overlap(self, box, current_intensity, scoring_params):
        return BoxExact.splitting_intensity_non_overlap(
            box = box,
            other_boxes = self._get_overlapping_boxes(box),
            current_intensity = current_intensity,
            scoring_params = scoring_params
        )

    def flexible_non_overlap(self, box, current_intensity, scoring_params):
        other_boxes = self._get_overlapping_boxes(box)
        areas = [box.overlap_raw(b) for b in other_boxes]
        non_overlapped = max(0.0, 1.0 - sum(areas) / box.area())
        norm_areas = [ar / box.area() for ar in areas]
        #NB: this seems wrong, but i tried to preserve it as it was - vinny?
        intensity_diff = [
            np.log(current_intensity) - np.log(max(1.0, b.intensity)) * na
            for b, na in zip(other_boxes, norm_areas)
        ]
        
        non_overlap = np.log(current_intensity) * non_overlapped
        refragment = sum(
            max(0.0, int_diff) for int_diff in intensity_diff
        )
        refragment2 = sum(intensity_diff)
        
        new_peak_score = sum(
            np.log(current_intensity) * na 
            for b, na in zip(other_boxes, norm_areas)
            if math.isclose(0.0, b.intensity)
        )
        
        return (
            non_overlap 
            + scoring_params["theta1"] * refragment 
            + scoring_params["theta2"] * refragment2 
            + scoring_params["theta3"] * new_peak_score
        )

    def case_control_non_overlap(self, box, current_intensity, scoring_params):
        other_boxes = self._get_overlapping_boxes(box)
        areas = [box.overlap_raw(b) for b in other_boxes]
        non_overlapped = max(0.0, 1.0 - sum(areas) / box.area())
        norm_areas = [ar / box.area() for ar in areas]
        #NB: this seems wrong, but i tried to preserve it as it was - vinny?
        intensity_diff = [
            np.log(current_intensity) - np.log(max(1.0, b.intensity)) * na
            for b, na in zip(other_boxes, norm_areas)
        ]
        
        non_overlap = np.log(current_intensity) * non_overlapped
        refragment = sum(
            max(0.0, int_diff) for int_diff in intensity_diff
        )
        refragment2 = sum(intensity_diff)
        
        new_peak_score = sum(
            np.log(current_intensity) * na 
            for b, na in zip(other_boxes, norm_areas)
            if math.isclose(0.0, b.intensity)
        )
        
        if box.pvalue is None:
            model_score = 0.0
        else:
            model_score = sum(
                max(0.0, np.log(current_intensity) - max(0.0, np.log(b.intensity)) * na)
                for b, na in zip(other_boxes, norm_areas)
            ) * (1 - box.pvalue)
                   
        return (
            non_overlap 
            + scoring_params["theta1"] * refragment 
            + scoring_params["theta2"] * refragment2 
            + scoring_params["theta3"] * new_peak_score
            + scoring_params["theta4"] * model_score
        )
        
            
class BoxGrid(BoxExact): 
    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size):
        self.grid = LocatorGrid(min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size)
        
    def get_all_boxes(self):
        return self.grid.get_all_boxes()
    
    def point_in_box(self, pt):
        box = GenericBox(pt.x, pt.x, pt.y, pt.y)
        return any(
            box.contains_point(pt) for b in self.grid.get_boxes(box)
        )
        
    def point_in_which_boxes(self, pt):
        box = GenericBox(pt.x, pt.x, pt.y, pt.y)
        return set(
            b for b in self.grid.get_boxes(box) if b.contains_point(pt)
        )
        
    def interval_overlaps_which_boxes(self, inv):
        box = GenericBox(inv.pt1.x, inv.pt2.x, inv.pt1.y, inv.pt2.y)
        return set(
            b for b in self.grid.get_boxes(box) if b.overlaps_with_box(box)
        )
        
    def interval_covers_which_boxes(self, inv):
        box = GenericBox(inv.pt1.x, inv.pt2.x, inv.pt1.y, inv.pt2.y)
        return set(
            b for b in self.grid.get_boxes(box) if b.interval_contains(inv)
        )
        
    def _get_overlapping_boxes(self, box):
        return set(
            b for b in self.grid.get_boxes(box) if box.overlaps_with_box(b)
        )
        
    def register_boxes(self, boxes):
        for b in boxes:
            self.grid.register_box(b)
        
    def clear(self):
        self.grid.clear()  
        
        
class BoxIntervalTrees(BoxExact):  
    def __init__(self):
        self.y_tree = intervaltree.IntervalTree()
        
    def get_all_boxes(self):
        return (
            inv.data for inv in self.y_tree.items()
        )
    
    def point_in_box(self, pt):
        return any(
            yb.data.pt1.x <= pt.x and yb.data.pt2.x >= pt.x
            for yb in self.y_tree.at(pt.y)
        )
        
    def point_in_which_boxes(self, pt):
        return set(
            yb.data
            for yb in self.y_tree.at(pt.y)
            if yb.data.pt1.x <= pt.x and yb.data.pt2.x >= pt.x
        )
        
    def interval_overlaps_which_boxes(self, inv):
        if(inv.is_vertical()):
            return set(
                yb.data 
                for yb in self.y_tree.overlap(inv.pt1.y, inv.pt2.y)
                if yb.data.pt1.x <= inv.pt1.x and yb.data.pt2.x >= inv.pt1.x
            )
        else:
            return set(
                yb.data
                for yb in self.y_tree.at(inv.pt1.y)
                if not (yb.data.pt1.x > inv.pt2.x or yb.data.pt2.x < inv.pt1.x)
            )
        
    def interval_covers_which_boxes(self, inv):
        if(inv.is_vertical()):
            return set(
                yb.data 
                for yb in self.y_tree.envelop(inv.pt1.y, inv.pt2.y)
                if yb.data.pt1.x <= inv.pt1.x and yb.data.pt2.x >= inv.pt1.x
            )
        else:
            return set(
                yb.data
                for yb in self.y_tree.at(inv.pt1.y)
                if yb.data.pt1.x <= inv.pt1.x and yb.data.pt2.x >= inv.pt2.x
            )
        
    def _get_overlapping_boxes(self, box):
        return set(
            yb.data 
            for yb in self.y_tree.overlap(box.pt1.y, box.pt2.y)
            if not (yb.data.pt1.x > box.pt2.x or yb.data.pt2.x < box.pt1.x)
        )
        
    def register_boxes(self, boxes):
        for b in boxes:
            self.y_tree.addi(b.pt1.y, b.pt2.y + 1E-12, b)
        
    def clear(self):
        self.__init__()

        
class BoxLineSweeper(BoxExact):
    def __init__(self):
        self.lswp = LineSweeper()
        self.running_scores = dict()
        
    def get_all_boxes(self):
        return self.lswp.get_all_boxes()
    
    def set_active_boxes(self, current_loc):
        self.lswp.set_active_boxes(current_loc)
        
    def point_in_box(self, pt):
        return self.lswp.point_in_box(pt)
        
    def point_in_which_boxes(self, pt):
        return self.lswp.point_in_which_boxes(pt)
        
    def interval_overlaps_which_boxes(self, inv):
        return self.lswp.interval_overlaps_which_boxes(inv)
        
    def interval_covers_which_boxes(self, inv):
        return self.lswp.interval_covers_which_boxes(inv)
        
    def _get_overlapping_boxes(self, box):
        raise NotImplementedError("Not used in this class: this line shouldn't be reached.")
        
    def non_overlap(self, box):
        '''NB: This won't work if the boxes are capable of moving between time updates.'''
        sliced = box.copy()
        sliced.pt1.x = self.lswp.previous_loc
        
        other_boxes = (
            self.lswp.interval_overlaps_which_boxes(
                Interval(
                    self.lswp.current_loc,
                    self.lswp.current_loc,
                    sliced.pt1.y,
                    sliced.pt2.y
                )
            ) 
            + self.lswp.was_active
        ) #TODO: If the manual intersection check is removed from this non-overlap, then filter to intersecting boxes
        sliced_uncovered = sum(b.area() for b in BoxExact.non_overlap_boxes(sliced, other_boxes))
        
        running_uncovered, running_total = self.running_scores.get(sliced.id, (0, 0))
        running_uncovered += sliced_uncovered
        running_total += sliced.area()
        self.running_scores[sliced.id] = (running_uncovered, running_total)
        
        return running_uncovered / running_total
        
    def intensity_non_overlap(self, box):
        raise NotImplementedError("BoxLineSweeper doesn't implement this yet!")
    
    def register_boxes(self, boxes):
        self.lswp.register_boxes(boxes)
        
    def clear(self):
        self.__init__()
    
    #TODO:
    #for evaluation
    #for point matching, find which precursors lie in which boxes, then check if fragmentation scans are still in box and have any as parent 
    #for window matching, find which precursors lie in which boxes, find which boxes are overlapped by which intervals, and use largest precursor (or sum of precursors?) in that box
    
    #for matching
    #find which precursors lie in which boxes with points from a single scan, then create edge with intensity at that point