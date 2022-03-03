from collections import deque

from vimms.Roi import RoiAligner


class GridEstimator():
    """Wrapper class letting internal grid be updated with rt drift
    estimates."""

    def __init__(self, grid, drift_model, min_rt_width=0.01,
                 min_mz_width=0.01):
        self.pending_ms2s = deque()
        self.observed_rois = [[]]
        self.grid = grid
        self.drift_models = [drift_model]
        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width
        self.injection_count = 0

    def non_overlap(self, box):
        return self.grid.non_overlap(box)

    def intensity_non_overlap(self, box, current_intensity, scoring_params):
        return self.grid.intensity_non_overlap(box, current_intensity,
                                               scoring_params)

    def flexible_non_overlap(self, box, current_intensity, scoring_params):
        return self.grid.flexible_non_overlap(box, current_intensity,
                                              scoring_params)

    def case_control_no_overlap(self, box, current_intensity, scoring_params):
        return self.grid.case_control_non_overlap(box, current_intensity,
                                                  scoring_params)

    def register_roi(self, roi):
        self.pending_ms2s.append(roi)

    def get_estimator(self):
        fn = self.drift_models[self.injection_count].get_estimator(
            self.injection_count)
        return lambda roi: fn(roi, self.injection_count)

    def _update_grid(self):
        self.grid.boxes = self.grid.init_boxes(self.grid.rtboxes,
                                               self.grid.mzboxes)
        for inj_num, inj in enumerate(self.observed_rois):
            fn = self.drift_models[inj_num].get_estimator(inj_num)
            for roi in inj:
                drift, _ = fn(roi, inj_num)
                self.grid.register_box(
                    roi.to_box(self.min_rt_width, self.min_mz_width,
                               rt_shift=(-drift)))

    def _next_model(self):
        self.observed_rois.append([])
        self.drift_models.append(self.drift_models[-1]._next_model())
        self.injection_count += 1

    def send_training_data(self, scan):
        if (scan.ms_level != 2):
            return
        roi = self.pending_ms2s.popleft()
        self.drift_models[-1].send_training_data(scan, roi,
                                                 self.injection_count)
        self.observed_rois[self.injection_count].append(roi)

    # TODO: later we could have arbitrary update points rather than after
    #  injection
    def update_after_injection(self):
        self._update_grid()
        self._next_model()


class CaseControlGridEstimator(GridEstimator):
    def __init__(self, grid, drift_model, min_rt_width=0.01, min_mz_width=0.01,
                 rt_tolerance=100, box_method='mean'):
        super().__init__(grid, drift_model, min_rt_width=min_rt_width,
                         min_mz_width=min_mz_width)
        self.rt_tolerance = rt_tolerance
        self.box_method = box_method

    def _update_grid(self):
        self.grid.boxes = self.grid.init_boxes(self.grid.rtboxes,
                                               self.grid.mzboxes)
        roi_aligner = RoiAligner(rt_tolerance=self.rt_tolerance)
        for inj_num, inj in enumerate(self.observed_rois):
            fn = self.drift_models[inj_num].get_estimator(inj_num)
            rt_shifts = [-fn(roi, inj_num)[0] for roi in inj]
            roi_aligner.add_sample(self.observed_rois, self.grid.sample_number,
                                   rt_shifts=rt_shifts)
        boxes = roi_aligner.get_boxes(
            method=self.box_method)  # TODO might need to add intensity here
        for box in boxes:
            self.grid.register_box(box)
