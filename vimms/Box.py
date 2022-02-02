import math
import bisect
import itertools
import random
from abc import abstractmethod
from collections import defaultdict, deque
from decimal import Decimal
from functools import reduce

import intervaltree
import GPy
import numpy as np
from mass_spec_utils.data_import.mzmine import PickedBox
from mass_spec_utils.library_matching.spectral_scoring_functions import cosine_similarity
from mass_spec_utils.library_matching.spectrum import Spectrum


class Point():
    def __init__(self, x, y): self.x, self.y = float(x), float(y)

    def __eq__(self, other_point): return math.isclose(self.x, other_point.x) and math.isclose(self.y, other_point.y)

    def __repr__(self): return "Point({}, {})".format(self.x, self.y)


class Box():
    def __init__(self, x1, x2, y1, y2, parents=None, min_xwidth=0, min_ywidth=0, intensity=0, id=None):
        self.id = id
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
        self.parents = [self] if parents is None else parents
        self.intensity = intensity

        if (self.pt2.x - self.pt1.x < min_xwidth):
            midpoint = self.pt1.x + ((self.pt2.x - self.pt1.x) / 2)
            self.pt1.x, self.pt2.x = midpoint - (min_xwidth / 2), midpoint + (min_xwidth / 2)

        if (self.pt2.y - self.pt1.y < min_ywidth):
            midpoint = self.pt1.y + ((self.pt2.y - self.pt1.y) / 2)
            self.pt1.y, self.pt2.y = midpoint - (min_ywidth / 2), midpoint + (min_ywidth / 2)

    def __repr__(self):
        return "Box({}, {})".format(self.pt1, self.pt2)

    def __eq__(self, other_box):
        return self.pt1 == other_box.pt1 and self.pt2 == other_box.pt2

    def __hash__(self):
        return (self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y).__hash__()

    def area(self):
        return (self.pt2.x - self.pt1.x) * (self.pt2.y - self.pt1.y)

    def copy(self, xshift=0, yshift=0):
        return type(self)(self.pt1.x + xshift, self.pt2.x + xshift, self.pt1.y + yshift, self.pt2.y + yshift,
                          parents=self.parents, intensity=self.intensity, id=self.id)

    def shift(self, xshift=0, yshift=0):
        self.pt1.x += xshift
        self.pt2.x += xshift
        self.pt1.y += yshift
        self.pt2.y += yshift

    def num_overlaps(self):
        return len(self.parents)

    def to_pickedbox(self, peak_id):
        rts = [self.pt1.x, self.pt2.x]
        mzs = [self.pt1.y, self.pt2.y]
        return PickedBox(peak_id, sum(mzs) / 2, sum(rts) / 2, mzs[0], mzs[1], rts[0], rts[1])


class GenericBox(Box):
    '''Makes no particular assumptions about bounding boxes.'''

    def __repr__(self):
        return "Generic{}".format(super().__repr__())

    def overlaps_with_box(self, other_box):
        return (self.pt1.x < other_box.pt2.x and self.pt2.x > other_box.pt1.x) and (
                    self.pt1.y < other_box.pt2.y and self.pt2.y > other_box.pt1.y)

    def contains_box(self, other_box):
        return (
                self.pt1.x <= other_box.pt1.x
                and self.pt1.y <= other_box.pt1.y
                and self.pt2.x >= other_box.pt2.x
                and self.pt2.y >= other_box.pt2.y
        )

    def overlap_2(self, other_box):
        if (not self.overlaps_with_box(other_box)): return 0.0
        b = Box(max(self.pt1.x, other_box.pt1.x), min(self.pt2.x, other_box.pt2.x), max(self.pt1.y, other_box.pt1.y),
                min(self.pt2.y, other_box.pt2.y))
        return b.area() / (self.area() + other_box.area() - b.area())

    def overlap_3(self, other_box):
        if (not self.overlaps_with_box(other_box)): return 0.0
        b = Box(max(self.pt1.x, other_box.pt1.x), min(self.pt2.x, other_box.pt2.x), max(self.pt1.y, other_box.pt1.y),
                min(self.pt2.y, other_box.pt2.y))
        return b.area() / self.area()

    def non_overlap_split(self, other_box):
        '''Finds 1 to 4 boxes describing the polygon of area of this box not overlapped by other_box.
           If one box is found, crops this box to dimensions of that box, and returns None.
           Otherwise, returns list of 2 to 4 boxes. Number of boxes found is equal to number of edges overlapping area does NOT share with this box.'''
        if (not self.overlaps_with_box(other_box)): return None
        x1, x2, y1, y2 = self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y
        split_boxes = []
        if (other_box.pt1.x > self.pt1.x):
            x1 = other_box.pt1.x
            split_boxes.append(GenericBox(self.pt1.x, x1, y1, y2, parents=self.parents, intensity=self.intensity))
        if (other_box.pt2.x < self.pt2.x):
            x2 = other_box.pt2.x
            split_boxes.append(GenericBox(x2, self.pt2.x, y1, y2, parents=self.parents, intensity=self.intensity))
        if (other_box.pt1.y > self.pt1.y):
            y1 = other_box.pt1.y
            split_boxes.append(GenericBox(x1, x2, self.pt1.y, y1, parents=self.parents, intensity=self.intensity))
        if (other_box.pt2.y < self.pt2.y):
            y2 = other_box.pt2.y
            split_boxes.append(GenericBox(x1, x2, y2, self.pt2.y, parents=self.parents, intensity=self.intensity))
        return split_boxes

    def split_all(self, other_box):
        if (not self.overlaps_with_box(other_box)): return None, None, None
        both_parents = self.parents + other_box.parents
        both_box = type(self)(max(self.pt1.x, other_box.pt1.x), min(self.pt2.x, other_box.pt2.x),
                              max(self.pt1.y, other_box.pt1.y), min(self.pt2.y, other_box.pt2.y), parents=both_parents,
                              intensity=max(self.intensity, other_box.intensity))
        b1_boxes = self.non_overlap_split(other_box)
        b2_boxes = other_box.non_overlap_split(self)
        return b1_boxes, b2_boxes, both_box


class Grid():

    @staticmethod
    @abstractmethod
    def init_boxes(): pass

    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.min_mz, self.max_mz = min_mz, max_mz
        self.rt_box_size, self.mz_box_size = rt_box_size, mz_box_size
        self.box_area = float(Decimal(rt_box_size) * Decimal(mz_box_size))

        self.rtboxes = range(0, int((self.max_rt - self.min_rt) / rt_box_size) + 1)
        self.mzboxes = range(0, int((self.max_mz - self.min_mz) / mz_box_size) + 1)
        self.boxes = self.init_boxes(self.rtboxes, self.mzboxes)

    def get_box_ranges(self, box):
        rt_box_range = (
        int((box.pt1.x - self.min_rt) / self.rt_box_size), int((box.pt2.x - self.min_rt) / self.rt_box_size) + 1)
        mz_box_range = (
        int((box.pt1.y - self.min_mz) / self.mz_box_size), int((box.pt2.y - self.min_mz) / self.mz_box_size) + 1)
        total_boxes = (rt_box_range[1] - rt_box_range[0]) * (mz_box_range[1] - mz_box_range[0])
        return rt_box_range, mz_box_range, total_boxes

    @abstractmethod
    def non_overlap(self, box): pass

    @abstractmethod
    def register_box(self, box): pass
    
    def register_boxes(self, boxes):
        for b in boxes: self.register_box(b)
        
    def set_active_boxes(self, *args):
        pass

    def clear(self):
        self.__init__(self.min_rt, self.max_rt, self.rt_box_size, self.min_mz, self.max_mz, self.mz_box_size)

class DictGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes):
        return defaultdict(list)

    def non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        return sum(
            float(not self.boxes[(rt, mz)]) for rt in range(*rt_box_range) for mz in range(*mz_box_range)) / total_boxes

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for rt in range(*rt_box_range):
            for mz in range(*mz_box_range):
                self.boxes[(rt, mz)].append(box)


class ArrayGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return np.array([[False for mz in mzboxes] for rt in rtboxes])

    def non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        boxes = self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]
        return (total_boxes - np.sum(boxes)) / total_boxes

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]] = True


class LocatorGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes):
        arr = np.empty((max(rtboxes), max(mzboxes)), dtype=object)
        for i, row in enumerate(arr):
            for j, _ in enumerate(row): arr[i, j] = set()
        return arr

    def get_boxes(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        boxes = set()
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for s in row: boxes |= s
        return boxes

    def all_boxes(self):
        return reduce(lambda s1, s2: s1 | s2, (s for row in self.boxes for s in row))

    @staticmethod
    def dummy_non_overlap(box, other_boxes):
        return 1.0
        
    @staticmethod
    def non_overlap_boxes(box, other_boxes):
        new_boxes = [box]
        for b in other_boxes:  # filter boxes down via grid with large boxes for this loop + boxes could be potentially sorted by size (O(n) insert time in worst-case)?
            if (box.overlaps_with_box(b)):  # quickly exits any box not overlapping new box
                updated_boxes = []
                for b2 in new_boxes:
                    if (not b.contains_box(
                            b2)):  # if your box is contained within a previous box area is 0 and box is not carried over
                        split_boxes = b2.non_overlap_split(b)
                        if (not split_boxes is None):
                            updated_boxes.extend(split_boxes)
                        else:
                            updated_boxes.append(b2)
                if (not updated_boxes): return []
                new_boxes = updated_boxes
        return new_boxes

    @staticmethod
    def splitting_non_overlap(box, other_boxes):
        boxes = LocatorGrid.non_overlap_boxes(box, other_boxes)
        if(boxes == []): return 0.0
        else: return sum(b.area() for b in boxes) / box.area()

    def non_overlap(self, box):
        return self.splitting_non_overlap(box, self.get_boxes(box))

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for s in row:
                s.add(box)


class AllOverlapGrid(LocatorGrid):
    @staticmethod
    def split_all_boxes(box, other_boxes):
        this_non, other_non, overlaps = [box], [], []
        for other in other_boxes:
            if (box.overlaps_with_box(other)):
                updated_this, others = [], [other]
                for i, this in enumerate(this_non):
                    if (others == []):
                        updated_this.extend(this_non[i:])
                        break
                    updated_others, split = [], False
                    for o in others:
                        this_bs, other_bs, both_b = this.split_all(o)
                        if (not both_b is None):
                            overlaps.append(both_b)
                            split = True
                            updated_this.extend(this_bs)
                            updated_others.extend(other_bs)
                        else:
                            updated_others.append(o)
                    if (not split): updated_this.append(this)
                    others = updated_others
                other_non.extend(others)
                this_non = updated_this
            else:
                other_non.append(other)
        return this_non, other_non, overlaps

    def intensity_non_overlap(self, box, current_intensity, scoring_params):
        box = box.copy()
        box.intensity = 0.0
        other_boxes = self.get_boxes(box)
        this_non, _, overlaps = self.split_all_boxes(box, other_boxes)
        non_overlap = current_intensity ** (sum(b.area() for b in this_non) / box.area())
        refragment = sum(max(0.0, current_intensity - b.intensity) ** (b.area() / box.area()) for b in overlaps)
        return non_overlap + scoring_params['theta1'] * refragment

    def flexible_non_overlap(self, box, current_intensity, scoring_params):
        box = box.copy()
        box.intensity = 0.0
        other_boxes = self.get_boxes(box)
        this_non, _, overlaps = self.split_all_boxes(box, other_boxes)
        non_overlap = np.log(current_intensity ** (sum(b.area() for b in this_non) / box.area()))
        refragment = scoring_params['theta1'] * sum(max(0.0, np.log(current_intensity) - np.log(max(1.0, b.intensity))
                                                    * b.area() / box.area()) for b in overlaps)
        refragment2 = scoring_params['theta2'] * sum(np.log(current_intensity) - np.log(max(1.0, b.intensity)) *
                                                     (b.area() / box.area()) for b in overlaps)
        new_peak = []
        for b in overlaps:
            if b.intensity == 0.0:
                new_peak.append(np.log(current_intensity) * (b.area() / box.area()))
        new_peak_score = scoring_params['theta3'] * sum(new_peak)
        return non_overlap + refragment + refragment2 + new_peak_score

    def case_control_non_overlap(self, box, current_intensity, scoring_params):
        box = box.copy()
        box.intensity = 0.0
        other_boxes = self.get_boxes(box)
        this_non, _, overlaps = self.split_all_boxes(box, other_boxes)
        non_overlap = np.log(current_intensity ** (sum(b.area() for b in this_non) / box.area()))
        refragment = scoring_params['theta1'] * sum(max(0.0, np.log(current_intensity) - max(0.0, np.log(b.intensity))
                                                    * b.area() / box.area()) for b in overlaps)
        refragment2 = scoring_params['theta2'] * sum(np.log(current_intensity) - max(0.0, np.log(b.intensity)) *
                                                     (b.area() / box.area()) for b in overlaps)
        new_peak = []
        for b in overlaps:
            if b.intensity == 0.0:
                new_peak.append(np.log(current_intensity) * (b.area() / box.area()))
        new_peak_score = scoring_params['theta3'] * sum(new_peak)
        if box.pvalue is None:
            return non_overlap + refragment + refragment2 + new_peak_score
        else:
            model_score = scoring_params['theta4'] * (1 - box.pvalue) * sum(max(0.0, np.log(current_intensity) -
                                            max(0.0, np.log(b.intensity)) * b.area() / box.area()) for b in overlaps)
            return non_overlap + refragment + refragment2 + new_peak_score + model_score

    def register_box(self, box):
        other_boxes = self.get_boxes(box)
        this_non, other_non, overlaps = self.split_all_boxes(box, other_boxes)
        for b in other_boxes:
            if (box.overlaps_with_box(b)):
                rt_box_range, mz_box_range, _ = self.get_box_ranges(b)
                for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
                    for s in row:
                        s.remove(b)

        for b in itertools.chain(this_non, other_non, overlaps):
            rt_box_range, mz_box_range, _ = self.get_box_ranges(b)
            for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
                for s in row:
                    s.add(b)

    def boxes_by_overlaps(self, boxes=None):
        binned, boxes = [], self.all_boxes() if boxes is None else reduce(
            lambda bs, b: [bx for t in self.split_all_boxes(b, bs) for bx in t], boxes, [])
        for b in boxes:
            while (len(binned) < b.num_overlaps()):
                binned.append([])
            binned[b.num_overlaps() - 1].append(b)
        return binned
        
class SweepGrid():
    # aiming for [1011, 1874, 2297, 2514, 2635, 2721]
    def __init__(self, track=True):
        self.all_boxes, self.active, self.was_active, self.active_ends = [], [], [], []
        self.active_intervals = intervaltree.IntervalTree()
        self.previous_time, self.current_time = -1, 0
        self.active_pointer = 0
        
        #self.running_scores = {}
        self.running_scores = defaultdict(list)
        
        #TODO: remove test code
        self.heights = defaultdict(list)
    
    def update_active(self, new_boxes):
        for b in new_boxes:
            end_point = b.pt2.x
            i = bisect.bisect_right(self.active_ends, -end_point) #invert value for descending sort
            self.active.insert(i, b)
            self.active_ends.insert(i, -end_point)
            self.active_intervals.addi(b.pt1.y, b.pt2.y, b)
    
    def set_active_boxes(self, current_time):
        if(current_time < self.current_time):
            self.active_pointer = 0
            self.active = []
            
        next_ptr = self.active_pointer
        while(next_ptr < len(self.all_boxes) and self.all_boxes[next_ptr].pt1.x <= current_time):
            next_ptr += 1
        self.update_active(self.all_boxes[self.active_pointer:next_ptr])
        self.active_pointer = next_ptr
        
        #TODO: remove test code
        assert len(self.active) == len(self.active_ends), f"length {len(self.active)} of active does not match length {len(self.active_ends)} of active_ends"
        import math
        for b, t in zip(self.active, self.active_ends):
            assert math.isclose(b.pt2.x, -t), f"time {b.pt2.x} of active does not match time {-t} of active_ends"
        assert len(self.active) == len(self.active_intervals), f"length {len(self.active)} of active does not match length {len(self.active_intervals)} of active_intervals"
        
        self.was_active = []
        while(self.active != [] and self.active[-1].pt2.x <= current_time):
            b = self.active.pop()
            self.active_ends.pop()
            self.was_active.append(b)
            self.active_intervals.removei(b.pt1.y, b.pt2.y, b)
        self.previous_time = self.current_time
        self.current_time = current_time
        
    def point_in_box(self, mz):
        return self.active_interval.overlaps_point(mz)
        
    def point_in_which_boxes(self, mz):
        return [inv.data for inv in self.active_intervals.at(mz)]
        
    def interval_overlaps_which_boxes(self, min_mz, max_mz):
        return [inv.data for inv in self.active_intervals.overlap(min_mz, max_mz)]
        
    def interval_covers_which_boxes(self, min_mz, max_mz):
        return [inv.data for inv in self.active_intervals.envelop(min_mz, max_mz)]
        
    def non_overlap(self, box):
        sliced = box.copy()
        sliced.pt1.x = self.previous_time
        
        other_boxes = self.interval_overlaps_which_boxes(sliced.pt1.y, sliced.pt2.y) + self.was_active #TODO: If the manual intersection check is removed from this non-overlap, then filter to intersecting boxes
        sliced_uncovered = sum(b.area() for b in LocatorGrid.non_overlap_boxes(sliced, other_boxes))
        
        #TODO: remove test code
        #height = sliced.pt2.y - sliced.pt1.y
        #for b, h in self.heights[sliced.id]:
        #    assert math.isclose(height, h), f"Height {height} of box {sliced} does not match previous height {h} of box {b}!"
        
        #TODO: remove test code
        #test_member = set(other_boxes)
        #for b in self.all_boxes:
        #    msg = f"box {b} not in all_boxes at prev time {self.previous_time} & curr time {self.current_time} with slice {sliced}"
        #    assert (not b.overlaps_with_box(sliced) 
        #            or b in test_member
        #            ), msg
        
        '''
        running_uncovered, running_total = self.running_scores.get(sliced.id, (0, 0))
        running_uncovered += sliced_uncovered
        running_total += sliced.area()
        self.running_scores[sliced.id] = (running_uncovered, running_total)
        '''
        
        '''
        self.running_scores[sliced.id].append((sliced_uncovered, sliced.area()))
        areas = np.sum(self.running_scores[sliced.id], axis=0)
        running_uncovered, running_total = areas
        '''
        
        from decimal import Decimal
        pair = (Decimal(sliced_uncovered), Decimal(sliced.area()))
        self.running_scores[sliced.id].append(pair)
        running_uncovered = sum(area for area, _ in self.running_scores[sliced.id])
        running_total = sum(area for _, area in self.running_scores[sliced.id])
        
        #TODO: remove test code
        self.heights[sliced.id].append((sliced, (sliced.pt2.y - sliced.pt1.y)))
        
        return float(running_uncovered / running_total)
        
    #TODO: another method to find intensity of boxes with max/sum option, provided some points
    
    def register_boxes(self, boxes):
        self.all_boxes = sorted(set(self.all_boxes + boxes), key=lambda b: b.pt1.x)
        
    def clear(self):
        self.__init__()
    
    #TODO:
    #for controller, we need to know which boxes are active, and which have (temporarily) ceased activity
    #using these, we can calculate e.g. non-overlap score using a small box slice, and keep a running count of all the scores of the slices so we can combine them
    #we still calculate slices for RoIs which are "inactive" but not "discarded", but they get a score weight of zero
    #we also need to search the mz via a similar sorted stack procedure (or interval tree)
    #for hard exclusion we just care if our points fall in any box, so can mark with boolean
    #for (intensity-)non-overlap, for each slice we need to return all boxes that intersect it on both dimensions
    
    #for evaluation
    #for point matching, find which precursors lie in which boxes, then check if fragmentation scans are still in box and have any as parent 
    #for window matching, find which precursors lie in which boxes, find which boxes are overlapped by which intervals, and use largest precursor (or sum of precursors?) in that box
    
    #for matching
    #find which precursors lie in which boxes with points from a single scan, then create edge with intensity at that point

    #refactor class to be more in line with previous ones

class DriftModel():
    @abstractmethod
    def get_estimator(self, injection_number): pass

    @abstractmethod
    def _next_model(self): pass

    def send_training_data(self, scan, roi, inj_num): pass

    def send_training_pair(self, x, y): pass

    def observed_points(self): return []

    def update(self, **kwargs): pass


class IdentityDrift(DriftModel):
    '''Dummy drift model which does nothing, for testing purposes.'''

    def get_estimator(self, injection_number): return lambda roi, inj_num: (0, {})

    def _next_model(self, **kwargs): return self


class OracleDrift(DriftModel):
    '''Drift model that cheats by being given a 'true' rt drift fn. for every injection in simulation, for testing purposes.'''

    def __init__(self, drift_fns):
        self.drift_fns = drift_fns

    def _next_model(self, **kwargs): return self

    def get_estimator(self, injection_number):
        if (type(self.drift_fns) == type([])): return self.drift_fns[injection_number]
        return self.drift_fns


class OraclePointMatcher():
    MODE_ALLPOINTS = 0
    MODE_RTENABLED = 1
    MODE_FRAGPAIRS = 2

    def __init__(self, chem_rts_by_injection, chemicals, max_points=None, mode=None):
        if (not max_points is None and max_points < len(chem_rts_by_injection[0])):
            idxes = random.sample([i for i, _ in enumerate(chem_rts_by_injection[0])], max_points)
            self.chem_rts_by_injection = [[sample[i] for i in idxes] for sample in chem_rts_by_injection]
        else:
            self.chem_rts_by_injection = chem_rts_by_injection
        self.chem_to_idx = {chem if chem.base_chemical is None else chem.base_chemical: idx for chem, idx in
                            zip(chemicals, range(len(chem_rts_by_injection[0])))}
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])
        self.available = [False] * len(self.chem_rts_by_injection[0])
        self.mode = OraclePointMatcher.MODE_FRAGPAIRS if mode is None else mode

    def _next_model(self):
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])

    def send_training_data(self, model, scan, roi, inj_num):
        if (self.mode == OraclePointMatcher.MODE_FRAGPAIRS):
            if (not scan.fragevent is None):
                parent_chem = scan.fragevent.chem if scan.fragevent.chem.base_chemical is None else scan.fragevent.chem.base_chemical
                if (parent_chem in self.chem_to_idx):
                    i = self.chem_to_idx[parent_chem]
                    if (inj_num == 0):
                        self.available[i] = True
                    elif (self.available[i] and self.not_sent[i]):
                        model.send_training_pair(self.chem_rts_by_injection[inj_num][i],
                                                 self.chem_rts_by_injection[0][i])
                        self.not_sent[i] = False
        else:
            if (self.mode == OraclePointMatcher.MODE_RTENABLED):
                enable = lambda y: scan.rt > y
            else:
                enable = lambda y: True

            for i, (y, x) in enumerate(zip(self.chem_rts_by_injection[inj_num], self.chem_rts_by_injection[0])):
                if (self.not_sent[i] and enable(y)):
                    model.send_training_pair(y, x)
                    self.not_sent[i] = False


class MS2PointMatcher():
    def __init__(self, min_score=0.9, mass_tol=0.2, min_match=1):
        self.ms2s = [[]]
        self.min_score, self.mass_tol, self.min_match = min_score, mass_tol, min_match

    def _next_model(self):
        self.ms2s[0] = [(rt, s, None) for rt, s, _ in self.ms2s[0]]
        self.ms2s.append([])

    def send_training_data(self, model, scan, roi, inj_num):
        # TODO: put some limitation on mz(/rt?) of boxes that can be matched
        spectrum = Spectrum(roi.get_mean_mz(), list(zip(scan.mzs, scan.intensities)))

        rt, _, __ = roi[0]
        if(inj_num > 0):
            if(len(self.ms2s[0]) > 0):
                original_idx, original_spectrum, score = -1, None, -1
                for i, (_, s, __) in enumerate(self.ms2s[0]):
                    current_score, _ = cosine_similarity(spectrum, s, self.mass_tol, self.min_match)
                    if (current_score > score):
                        original_idx, original_spectrum, score = i, s, current_score
                if (score < self.min_score): return
                original_rt, original_scan, prev_match = self.ms2s[0][original_idx]
                # if(not prev_match is None and score > prev_match[1]): update previous match somehow
                self.ms2s[0][original_idx] = (original_rt, original_spectrum, (spectrum, score))
                self.ms2s[inj_num].append((rt, spectrum, None))
                model.send_training_pair(rt, original_rt)
        else:
            self.ms2s[0].append((rt, spectrum, None))


class GPDrift(DriftModel):
    '''Drift model that uses a Gaussian Process and known training points to learn a drift function with reference to points in the first injection.'''

    def __init__(self, kernel, point_matcher, max_points=None):
        self.kernel = kernel
        self.point_matcher = point_matcher
        self.Y, self.X = [], []
        self.model = None
        self.max_points = max_points

    # TODO: Ideally this would use _online_ learning rather than retraining the whole model every time...
    def get_estimator(self, injection_number):
        if (injection_number == 0 or self.Y == []):
            return lambda roi, inj_num: (0, {})
        else:
            if (self.model is None):
                if (self.max_points is None or self.max_points >= len(self.Y)):
                    Y, X = self.Y, self.X
                else:
                    Y, X = self.Y[-self.max_points:], self.X[-self.max_points:]
                self.model = GPy.models.GPRegression(np.array(Y).reshape((len(Y), 1)), np.array(X).reshape((len(X), 1)),
                                                     kernel=self.kernel)
                self.model.optimize()

            def predict(roi, inj_num):
                mean, variance = self.model.predict(np.array(roi[0][0]).reshape((1, 1)))
                return roi[0][0] - mean[0, 0], {"variance" : variance[0, 0]}
            return predict

    def _next_model(self, **kwargs):
        Y, X = kwargs.get("Y", []), kwargs.get("X", [])
        new_model = GPDrift(self.kernel.copy(), self.point_matcher, max_points=self.max_points)
        self.point_matcher._next_model()
        new_model.Y, new_model.X = Y, X
        return new_model

    def send_training_data(self, scan, roi, inj_num):
        self.point_matcher.send_training_data(self, scan, roi, inj_num)

    # TODO: update to allow updating points: search for point with matching x point then change corresponding y value
    def send_training_pair(self, y, x):
        self.Y.append(y)
        self.X.append(x)
        self.model = None

    def observed_points(self):
        return self.Y

    def update(self, **kwargs):
        Y, X = kwargs.get("Y", []), kwargs.get("X", [])


