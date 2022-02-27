from collections import deque
from operator import attrgetter

from vimms.Roi import Roi, RoiAligner
from vimms.Box import BoxGrid, BoxIntervalTrees, LineSweeper
from vimms.DriftCorrection import IdentityDrift

class BoxManager():

    nosplit_errmsg = (
        "Intensity Non-Overlap can't be used without splitting the boxes! "
        "Please set the BoxManager's box_splitter to be an instance of BoxSplitter."
    )

    def __init__(self, box_geometry=None,
                    box_converter=None,
                    box_splitter=None,
                    drift_model=None
                ):
                
        self.pending_ms2s = deque()
        self.observed_rois = [[]]
        self.boxes = [[]]
        self.injection_count = 0
        
        self.drift_models = [IdentityDrift()] if drift_model is None else [drift_model]
        self.box_converter = BoxConverterUnique() if box_converter is None else box_converter
        self.box_splitter = BoxNoSplitter() if box_splitter is None else box_splitter
        self.box_geometry = BoxIntervalTrees() if box_geometry is None else box_geometry

    def point_in_box(self, pt):
        return self.box_geometry.point_in_box(pt)
    
    def point_in_which_boxes(self, pt):
        return self.box_geometry.point_in_which_boxes(pt)
    
    def interval_overlaps_which_boxes(self, inv):
        return self.box_geometry.interval_overlaps_which_boxes(inv)
    
    def interval_covers_which_boxes(self, inv):
        return self.box_geometry.interval_covers_which_boxes(inv)

    def non_overlap(self, box):
        return self.box_geometry.non_overlap(box)

    def intensity_non_overlap(self, box, current_intensity, scoring_params):
        if(not isinstance(self.box_splitter, BoxSplitter)): 
            raise ValueError(self.nosplit_errmsg)
        return self.box_geometry.intensity_non_overlap(box, current_intensity, scoring_params)

    def flexible_non_overlap(self, box, current_intensity, scoring_params):
        errmsg = (
            "Some better exception should be (sometimes) thrown here "
            "if the BoxManager isn't set up properly!"
        )
        raise NotImplementedError(errmsg)
        return self.box_geometry.flexible_non_overlap(box, current_intensity, scoring_params)

    def case_control_non_overlap(self, box, current_intensity, scoring_params):
        errmsg = (
            "Some better exception should be (sometimes) thrown here "
            "if the BoxManager isn't set up properly!"
        )
        raise NotImplementedError(errmsg)
        return self.box_geometry.case_control_non_overlap(box, current_intensity, scoring_params)
        
    def register_roi(self, roi):
        self.pending_ms2s.append(roi)
        
    def register_box(self, box):
        self.boxes[self.injection_count].append(box)

    def get_estimator(self):
        fn = self.drift_models[self.injection_count].get_estimator(self.injection_count)
        return lambda roi: fn(roi, self.injection_count)
        
    def _next_model(self):
        self.observed_rois.append([])
        self.boxes.append([])
        self.drift_models.append(self.drift_models[-1]._next_model())
        self.injection_count += 1

    def send_training_data(self, scan):
        if (scan.ms_level != 2): return
        roi = self.pending_ms2s.popleft()
        self.drift_models[-1].send_training_data(scan, roi, self.injection_count)
        self.observed_rois[self.injection_count].append(roi)
        
    def set_active_boxes(self, current_rt):
        self.box_geometry.set_active_boxes(current_rt)
        
    def _update_geometry(self):
        self.box_geometry.clear()
        
        all_boxes = list()
        for inj_num, inj in enumerate(self.boxes):
            fn = self.drift_models[inj_num].get_estimator(inj_num)
            drifts = (fn(b.roi, inj_num)[0] for b in inj)
            all_boxes.extend([b.copy(xshift=(-drift)) for b, drift in zip(inj, drifts)])
        
        split_boxes = self.box_splitter.split_boxes(all_boxes)
        self.box_geometry.register_boxes(split_boxes)

    # TODO: later we could have arbitrary drift update points rather than after injection
    def update_after_injection(self):
        boxes = self.box_converter.rois2boxes(self.observed_rois[self.injection_count])
        self.boxes[self.injection_count].extend(boxes)
        self._update_geometry()
        self._next_model()
        
class BoxConverterIgnore():
    def rois2boxes(self, rois):
        return []
        
class BoxConverter():
    def __init__(self, min_rt_width=0.000001, min_mz_width=0.000001):
        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        
    def rois2boxes(self, rois):
        return [r.to_box(self.min_rt_width, self.min_mz_width) for r in rois]

class BoxConverterUnique(BoxConverter):
    def rois2boxes(self, rois):
        boxes_by_id = dict()
        
        for r in rois:
            b = r.to_box(self.min_rt_width, self.min_mz_width)
            if(b.id in boxes_by_id):
                boxes_by_id[b.id] = max(b, boxes_by_id[b.id], key=lambda b: b.area())
            else:
                boxes_by_id[b.id] = b
        
        return [b for _, b in boxes_by_id.items()]

class BoxNoSplitter():
    def split_boxes(self, boxes):
        return boxes

class BoxSplitter():
    def split_boxes(self, boxes):
        splitter = LineSweeper()
        splitter.register_boxes(boxes)
        return splitter.reduce_all_boxes()

#TODO: This could probably be implemented as a component later
'''
class CaseControlGridEstimator(GridEstimator):
    def __init__(self, grid, drift_model, min_rt_width=0.01, min_mz_width=0.01, rt_tolerance=100, box_method='mean'):
        super().__init__(grid, drift_model, min_rt_width=min_rt_width, min_mz_width=min_mz_width)
        self.rt_tolerance = rt_tolerance
        self.box_method = box_method

    def _update_grid(self):
        self.grid.boxes = self.grid.init_boxes(self.grid.rtboxes, self.grid.mzboxes)
        roi_aligner = RoiAligner(rt_tolerance=self.rt_tolerance)
        for inj_num, inj in enumerate(self.observed_rois):
            fn = self.drift_models[inj_num].get_estimator(inj_num)
            rt_shifts = [-fn(roi, inj_num)[0] for roi in inj]
            roi_aligner.add_sample(self.observed_rois, self.grid.sample_number, rt_shifts=rt_shifts)
        boxes = roi_aligner.get_boxes(method=self.box_method)  # TODO might need to add intensity here
        for box in boxes:
            self.grid.register_box(box)
'''

