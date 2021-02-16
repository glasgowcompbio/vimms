import math
import random
import itertools
import numpy as np
from decimal import Decimal
from collections import defaultdict
from abc import ABC, abstractmethod

from mass_spec_utils.data_import.mzmine import PickedBox
import GPy

class Point():
    def __init__(self, x, y): self.x, self.y = float(x), float(y)
    def __repr__(self): return "Point({}, {})".format(self.x, self.y)

class Box():
    def __init__(self, x1, x2, y1, y2, parents=[], min_xwidth=0, min_ywidth=0):
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
        self.parents = parents
        
        if(self.pt2.x - self.pt1.x < min_xwidth):
            midpoint = self.pt1.x + ((self.pt2.x - self.pt1.x) / 2)
            self.pt1.x, self.pt2.x = midpoint - (min_xwidth / 2), midpoint + (min_xwidth / 2)

        if(self.pt2.y - self.pt1.y < min_ywidth):
            midpoint = self.pt1.y + ((self.pt2.y - self.pt1.y) / 2)
            self.pt1.y, self.pt2.y = midpoint - (min_ywidth / 2), midpoint + (min_ywidth / 2)
        
    def __repr__(self): return "Box({}, {})".format(self.pt1, self.pt2)
    def __eq__(self, other_box): return self.pt1 == other_box.pt1 and self.pt2 == other_box.pt2
    def __hash__(self): return (self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y).__hash__()
    def area(self): return (self.pt2.x - self.pt1.x) * (self.pt2.y - self.pt1.y)
    def copy(self, xshift=0, yshift=0): return type(self)(self.pt1.x + xshift, self.pt2.x + xshift, self.pt1.y + yshift, self.pt2.y + yshift)
    def shift(self, xshift=0, yshift=0):
        self.pt1.x += xshift
        self.pt2.x += xshift
        self.pt1.y += yshift
        self.pt2.y += yshift
    def num_overlaps(self): return 1 if len(self.parents) == 0 else len(self.parents)
    def top_level_boxes(self): return [self.copy()] if self.parents == [] else self.parents

    def to_pickedbox(self, peak_id):
        rts = [self.pt1.x, self.pt2.x]
        mzs = [self.pt1.y, self.pt2.y]
        return PickedBox(peak_id, sum(mzs)/2, sum(rts)/2, mzs[0], mzs[1], rts[0], rts[1])


class GenericBox(Box):
    '''Makes no particular assumptions about bounding boxes.'''
    
    def __repr__(self): return "Generic{}".format(super().__repr__())
    
    def overlaps_with_box(self, other_box):
        return (self.pt1.x < other_box.pt2.x and self.pt2.x > other_box.pt1.x) and (self.pt1.y < other_box.pt2.y and self.pt2.y > other_box.pt1.y)
    
    def contains_box(self, other_box):
        return (
                self.pt1.x <= other_box.pt1.x 
                and self.pt1.y <= other_box.pt1.y 
                and self.pt2.x >= other_box.pt2.x 
                and self.pt2.y >= other_box.pt2.y
               )
               
    def overlap_2(self, other_box):
        if(not self.overlaps_with_box(other_box)): return 0.0
        b = Box(max(self.pt1.x, other_box.pt1.x), min(self.pt2.x, other_box.pt2.x), max(self.pt1.y, other_box.pt1.y), min(self.pt2.y, other_box.pt2.y))
        return b.area() / (self.area() + other_box.area() - b.area())
               
    def non_overlap_split(self, other_box):
        '''Finds 1 to 4 boxes describing the polygon of area of this box not overlapped by other_box.
           If one box is found, crops this box to dimensions of that box, and returns None.
           Otherwise, returns list of 2 to 4 boxes. Number of boxes found is equal to number of edges overlapping area does NOT share with this box.'''
        if(not self.overlaps_with_box(other_box)): return None
        x1, x2, y1, y2 = self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y
        split_boxes = []
        if(other_box.pt1.x > self.pt1.x):
            x1 = other_box.pt1.x
            split_boxes.append(GenericBox(self.pt1.x, x1, y1, y2, parents=self.parents))
        if(other_box.pt2.x < self.pt2.x):
            x2 = other_box.pt2.x
            split_boxes.append(GenericBox(x2, self.pt2.x, y1, y2, parents=self.parents))
        if(other_box.pt1.y > self.pt1.y):
            y1 = other_box.pt1.y
            split_boxes.append(GenericBox(x1, x2, self.pt1.y, y1, parents=self.parents))
        if(other_box.pt2.y < self.pt2.y):
            y2 = other_box.pt2.y
            split_boxes.append(GenericBox(x1, x2, y2, self.pt2.y, parents=self.parents))
        return split_boxes
        
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
        rt_box_range = (int(box.pt1.x / self.rt_box_size), int(box.pt2.x / self.rt_box_size) + 1)
        mz_box_range = (int(box.pt1.y / self.mz_box_size), int(box.pt2.y / self.mz_box_size) + 1)
        total_boxes = (rt_box_range[1] - rt_box_range[0]) * (mz_box_range[1] - mz_box_range[0])
        return rt_box_range, mz_box_range, total_boxes

    @abstractmethod
    def non_overlap(self, box): pass
        
    @abstractmethod
    def register_box(self, box): pass   
        
class DictGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return defaultdict(list)
    
    def non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        return sum(float(not self.boxes[(rt, mz)]) for rt in range(*rt_box_range) for mz in range(*mz_box_range)) / total_boxes
        
    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for rt in range(*rt_box_range):
            for mz in range(*mz_box_range): self.boxes[(rt, mz)].append(box)
    
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
            for j, _ in enumerate(row): arr[i, j] = list() 
        return arr
    
    def get_boxes(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        boxes = []
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for ls in row: boxes.append(ls)
        return boxes
        
    @staticmethod
    def dummy_non_overlap(box, *other_boxes): return 1.0   
        
    @staticmethod
    def splitting_non_overlap(box, *other_boxes):
        new_boxes = [box]
        for b in other_boxes: #filter boxes down via grid with large boxes for this loop + boxes could be potentially sorted by size (O(n) insert time in worst-case)?
            if(box.overlaps_with_box(b)): #quickly exits any box not overlapping new box
                updated_boxes = []
                for b2 in new_boxes:
                    if(not b.contains_box(b2)): #if your box is contained within a previous box area is 0 and box is not carried over
                        split_boxes = b2.non_overlap_split(b)
                        if(not split_boxes is None): updated_boxes.extend(split_boxes)
                        else: updated_boxes.append(b2)
                if(not updated_boxes): return 0.0
                new_boxes = updated_boxes
        return sum(b.area() for b in new_boxes) / box.area()
        
    def non_overlap(self, box):
        return self.splitting_non_overlap(box, *itertools.chain(*self.get_boxes(box)))

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for ls in row: ls.append(box)

class DriftModel():
    @abstractmethod
    def get_estimator(self, injection_number): pass
    @abstractmethod
    def _next_model(self): pass
    def send_training_data(self, scan, inj_num): pass
    def send_training_pair(self, x, y): pass
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
        if(type(self.drift_fns) == type([])): return self.drift_fns[injection_number]
        return self.drift_fns
        
class OraclePointMatcher():
    MODE_ALLPOINTS = 0
    MODE_RTENABLED = 1
    MODE_FRAGPAIRS = 2

    def __init__(self, chem_rts_by_injection, chemicals, max_points=None, mode=None):
        if(not max_points is None and max_points < len(chem_rts_by_injection[0])):
            idxes = random.sample([i for i, _ in enumerate(chem_rts_by_injection[0])], max_points)
            self.chem_rts_by_injection = [[sample[i] for i in idxes] for sample in chem_rts_by_injection]
        else: 
            self.chem_rts_by_injection = chem_rts_by_injection
        self.chem_to_idx = {chem if chem.base_chemical is None else chem.base_chemical : idx for chem, idx in zip(chemicals, range(len(chem_rts_by_injection[0])))}
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])
        self.available = [False] * len(self.chem_rts_by_injection[0])
        self.mode = OraclePointMatcher.MODE_FRAGPAIRS if mode is None else mode
        
    def _next_model(self):
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])

    def send_training_data(self, model, scan, inj_num):
        if(self.mode == OraclePointMatcher.MODE_FRAGPAIRS and not scan.fragevent is None):
            parent_chem = scan.fragevent.chem if scan.fragevent.chem.base_chemical is None else scan.fragevent.chem.base_chemical
            if(parent_chem in self.chem_to_idx):
                i = self.chem_to_idx[parent_chem]
                if(inj_num == 0): self.available[i] = True
                elif(self.available[i] and self.not_sent[i]):
                    model.send_training_pair(Y[i], X[i])
                    self.not_sent[i] = False
        else:
            if(self.mode == OraclePointMatcher.MODE_RTENABLED): enable = lambda y: scan.rt > y
            else: enable = lambda y: True
            
            for i, (y, x) in enumerate(zip(self.chem_rts_by_injection[inj_num], self.chem_rts_by_injection[0])):
                if(self.not_sent[i] and enable(y)):
                    model.send_training_pair(y, x)
                    self.not_sent[i] = False
        
class GPDrift(DriftModel):
    '''Drift model that uses a Gaussian Process and known training points to learn a drift function with reference to points in the first injection.'''
    def __init__(self, kernel, point_matcher, max_points=None):
        self.kernel = kernel
        self.point_matcher = point_matcher
        self.Y, self.X = [], []
        self.model = None
        self.max_points = max_points
        
    #TODO: Ideally this would use _online_ learning rather than retraining the whole model every time...
    def get_estimator(self, injection_number): 
        if(injection_number == 0 or self.Y == []): return lambda roi, inj_num: (0, {})
        else: 
            def predict(roi, inj_num):
                if(self.model is None):
                    if(self.max_points is None or self.max_points >= len(self.Y)): Y, X = self.Y, self.X
                    else: Y, X = self.Y[-self.max_points:], self.X[-self.max_points:]
                    self.model = GPy.models.GPRegression(np.array(Y).reshape((len(Y), 1)), np.array(X).reshape((len(X), 1)), kernel=self.kernel)
                    self.model.optimize()
                mean, variance = self.model.predict(np.array(roi.estimate_apex()).reshape((1, 1)))
                return roi.estimate_apex() - mean[0], {"variance" : variance[0]}
            return predict
        
    def _next_model(self, **kwargs):
        Y, X = kwargs.get("Y", []), kwargs.get("X", [])
        new_model = GPDrift(self.kernel, self.point_matcher, max_points=self.max_points)
        self.point_matcher._next_model()
        new_model.Y, new_model.X = Y, X
        return new_model
        
    def send_training_data(self, scan, inj_num): self.point_matcher.send_training_data(self, scan, inj_num)
        
    def send_training_pair(self, y, x):
        self.Y.append(y)
        self.X.append(x)
        self.model = None
        
    def update(self, **kwargs): Y, X = kwargs.get("Y", []), kwargs.get("X", [])
    
    #X is RoIs from first injection
    #Y is RoIs from injection we are currently at
    #we need some way of matching Y -> X in training points
    #then we can learn general drift fn. for points not seen in Y
    
    #receive every point we fragment
    #create matched pairs of ms1, ms2 spectra
    #return uncertainty
    #uncertainty used to acquire new matchable points
            
class GridEstimator():
    '''Wrapper class letting internal grid be updated with rt drift estimates.'''

    def __init__(self, grid, drift_model, min_rt_width=0.01, min_mz_width=0.01):
        self.observed_rois = [[]]
        self.grid = grid
        self.drift_models = [drift_model]
        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        self.injection_count = 0
    
    def non_overlap(self, box): return self.grid.non_overlap(box)
    def register_roi(self, roi): self.observed_rois[self.injection_count].append(roi)
    def get_estimator(self):
        fn = self.drift_models[self.injection_count].get_estimator(self.injection_count)
        return lambda roi: fn(roi, self.injection_count)
    
    def _update_grid(self):
        self.grid.boxes = self.grid.init_boxes(self.grid.rtboxes, self.grid.mzboxes)
        for inj_num, inj in enumerate(self.observed_rois):
            fn = self.drift_models[inj_num].get_estimator(inj_num)
            for roi in inj:
                drift = fn(roi, inj_num)[0]
                self.grid.register_box(roi.to_box(self.min_rt_width, self.min_mz_width, rt_shift=(-drift)))
    
    def _next_model(self):
        self.injection_count += 1
        self.observed_rois.append([])
        self.drift_models.append(self.drift_models[-1]._next_model())
        
    def send_training_data(self, scan): self.drift_models[-1].send_training_data(scan, self.injection_count)
    
    #TODO: later we could have arbitrary update points rather than after injection
    def update_after_injection(self):
        self._update_grid()
        self._next_model()