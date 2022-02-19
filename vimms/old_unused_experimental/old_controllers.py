# flake8: noqa

# class OptimalTopNController(TopNController):
#     def __init__(self, ionisation_mode, N,
#                  isolation_widths, mz_tols, rt_tols, min_ms1_intensity, box_file, score_method='intensity',
#                  params=None):
#         super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_shift=0,
#                          params=params)
#         if type(box_file) == str:
#             self.box_file = box_file
#             self._load_boxes()
#         else:
#             self.boxes = box_file
#
#         self.score_method = score_method
#
#     def _load_boxes(self):
#         self.boxes = load_picked_boxes(self.box_file)
#         logger.debug("Loaded {} boxes".format(len(self.boxes)))
#
#     def _process_scan(self, scan):
#         # if there's a previous ms1 scan to process
#         new_tasks = []
#         ms2_tasks = []
#         if self.scan_to_process is not None:
#
#             mzs = self.scan_to_process.mzs
#             intensities = self.scan_to_process.intensities
#             rt = self.scan_to_process.rt
#
#             # Find boxes that span the current rt value
#             sub_boxes = list(
#                 filter(lambda x: x.rt_range_in_seconds[0] <= rt and x.rt_range_in_seconds[1] >= rt, self.boxes))
#             mzi = zip(mzs, intensities)
#             # remove any peaks below min intensity
#             mzi = list(filter(lambda x: x[1] >= self.min_ms1_intensity, mzi))
#             # sort by mz for matching with the boxes
#             mzi.sort(key=lambda x: x[0])
#             sub_boxes.sort(key=lambda x: x.mz_range[0])
#             mzib = self._box_match(mzi, sub_boxes)  # (mz,intensity,box)
#
#             # If there are things to fragment, schedule the scans...
#             if len(mzib) > 0:
#                 # compute the scores
#                 mzibs = self._score_peak_boxes(mzib, rt, score=self.score_method)
#                 # loop over points in decreasing score
#                 fragmented_count = 0
#                 # idx = np.argsort(intensities)[::-1]
#                 mzs, intensities, matched_boxes, scores = zip(*mzibs)
#                 idx = np.argsort(scores)[::-1]
#
#                 for i in idx:
#                     mz = mzs[i]
#                     intensity = intensities[i]
#                     matched_box = matched_boxes[i]
#
#                     # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
#                     if fragmented_count >= self.N:
#                         logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
#                         break
#
#                     # create a new ms2 scan parameter to be sent to the mass spec
#                     precursor_scan_id = self.scan_to_process.scan_id
#                     dda_scan_params = self.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
#                                                                self.mz_tol, self.rt_tol)
#                     new_tasks.append(dda_scan_params)
#                     ms2_tasks.append(dda_scan_params)
#                     fragmented_count += 1
#                     self.current_task_id += 1
#
#                     pos = self.boxes.index(matched_box)
#                     del self.boxes[pos]
#
#             # an MS1 is added here, as we no longer send MS1s as default
#             ms1_scan_params = self.get_ms1_scan_params()
#             self.current_task_id += 1
#             self.next_processed_scan_id = self.current_task_id
#             new_tasks.append(ms1_scan_params)
#
#             # set this ms1 scan as has been processed
#             self.scan_to_process = None
#         return new_tasks
#
#     def _box_match(self, mzi, boxes):
#         # mzi and boxes are sorted by mz
#         # loop over the mzi
#         mzib = []
#         lower_boxes = [b.mz_range[0] for b in boxes]
#         for this_mzi in mzi:
#             # find the possible boxes
#             left_pos = bisect.bisect_right(lower_boxes, this_mzi[0])
#             if left_pos < len(boxes):
#                 left_pos -= 1  # this is the first possible box
#                 if left_pos == -1:  # peak is lower in mz than all boxes
#                     continue
#                 if this_mzi[0] < boxes[left_pos].mz_range[1]:
#                     # found a match
#                     # compute time proportion left in the peak
#                     matching_box = boxes[left_pos]
#                     mzib.append((this_mzi[0], this_mzi[1], matching_box))
#                     del boxes[left_pos]
#                     del lower_boxes[left_pos]
#                 else:
#                     # no match found
#                     pass
#             else:
#                 # no match found
#                 pass
#         return mzib
#
#     def _score_peak_boxes(self, mzib, current_rt, score='intensity'):
#         # mzib = (mz,intensity,box) tuple
#         if score == 'intensity':
#             # simplest: score = intensity
#             scores = [(mz, i, b, i) for mz, i, b in mzib]
#         elif score == 'urgency':
#             scores = [(mz, i, b, -(b.rt_range_in_seconds[1] - current_rt)) for mz, i, b in mzib]
#         elif score == 'apex':
#             scores = [(mz, i, b, current_rt - b.rt_in_seconds) for mz, i, b in mzib]
#         elif score == 'random':
#             scores = [(mz, i, b, np.random.rand()) for mz, i, b in mzib]
#         return scores

# class PurityController(TopNController):
#     def __init__(self, ionisation_mode, N, scan_param_changepoints,
#                  isolation_widths, mz_tols, rt_tols, min_ms1_intensity,
#                  n_purity_scans=None, purity_shift=None, purity_threshold=0, purity_randomise=True,
#                  purity_add_ms1=True, ms1_shift=0,
#                  params=None):
#         super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_shift=ms1_shift,
#                          params=params)
#
#         # make sure these are stored as numpy arrays
#         self.N = np.array(N)
#         self.isolation_width = np.array(isolation_widths)  # the isolation window (in Dalton) to select a precursor ion
#         self.mz_tols = np.array(
#             mz_tols)  # the m/z window (ppm) to prevent the same precursor ion to be fragmented again
#         self.rt_tols = np.array(rt_tols)  # the rt window to prevent the same precursor ion to be fragmented again
#         if scan_param_changepoints is not None:
#             self.scan_param_changepoints = np.array([0] + scan_param_changepoints)
#         else:
#             self.scan_param_changepoints = np.array([0])
#
#         # purity stuff
#         self.n_purity_scans = n_purity_scans
#         self.purity_shift = purity_shift
#         self.purity_threshold = purity_threshold
#         self.purity_randomise = purity_randomise
#         self.purity_add_ms1 = purity_add_ms1
#
#         # make sure the input are all correct
#         assert len(self.N) == len(self.scan_param_changepoints) == len(self.isolation_width) == len(
#             self.mz_tols) == len(self.rt_tols)
#         if self.purity_threshold != 0:
#             assert all(self.n_purity_scans <= np.array(self.N))
#
#     def _process_scan(self, scan):
#         # if there's a previous ms1 scan to process
#         new_tasks = []
#         if self.scan_to_process is not None:
#             # check queue size because we want to schedule both ms1 and ms2 in the hybrid controller
#
#             mzs = self.scan_to_process.mzs
#             intensities = self.scan_to_process.intensities
#             rt = self.scan_to_process.rt
#
#             # set up current scan parameters
#             current_N, current_rt_tol, idx = self._get_current_N_DEW(rt)
#             current_isolation_width = self.isolation_width[idx]
#             current_mz_tol = self.mz_tols[idx]
#
#             # calculate purities
#             purities = []
#             for mz_idx in range(len(self.scan_to_process.mzs)):
#                 nearby_mzs_idx = np.where(
#                     abs(self.scan_to_process.mzs - self.scan_to_process.mzs[mz_idx]) < current_isolation_width)
#                 if len(nearby_mzs_idx[0]) == 1:
#                     purities.append(1)
#                 else:
#                     total_intensity = sum(self.scan_to_process.intensities[nearby_mzs_idx])
#                     purities.append(self.scan_to_process.intensities[mz_idx] / total_intensity)
#
#             # loop over points in decreasing intensity
#             fragmented_count = 0
#             idx = np.argsort(intensities)[::-1]
#             ms2_tasks = []
#             for i in idx:
#                 mz = mzs[i]
#                 intensity = intensities[i]
#                 purity = purities[i]
#
#                 # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
#                 if fragmented_count >= current_N:
#                     logger.debug('Top-%d ions have been selected' % (current_N))
#                     break
#
#                 if intensity < self.min_ms1_intensity:
#                     logger.debug(
#                         'Minimum intensity threshold %f reached at %f, %d' % (
#                             self.min_ms1_intensity, intensity, fragmented_count))
#                     break
#
#                 # skip ion in the dynamic exclusion list of the mass spec
#                 if self._is_excluded(mz, rt):
#                     continue
#
#                 if purity <= self.purity_threshold:
#                     purity_shift_amounts = [self.purity_shift * (i - (self.n_purity_scans - 1) / 2) for i in
#                                             range(self.n_purity_scans)]
#                     if self.purity_randomise:
#                         purity_randomise_idx = np.random.choice(self.n_purity_scans, self.n_purity_scans, replace=False)
#                     else:
#                         purity_randomise_idx = range(self.n_purity_scans)
#                     for purity_idx in purity_randomise_idx:
#                         # create a new ms2 scan parameter to be sent to the mass spec
#                         precursor_scan_id = self.scan_to_process.scan_id
#                         dda_scan_params = self.get_ms2_scan_params(mz + purity_shift_amounts[purity_idx],
#                                                                    intensity, precursor_scan_id,
#                                                                    current_isolation_width,
#                                                                    current_mz_tol,
#                                                                    current_rt_tol)
#                         new_tasks.append(dda_scan_params)
#                         ms2_tasks.append(dda_scan_params)
#                         self.current_task_id += 1
#                         if self.purity_add_ms1 and purity_idx != purity_randomise_idx[-1]:
#                             ms1_scan_params = get_default_scan_params(
#                                 polarity=self.environment.mass_spec.ionisation_mode)
#                             new_tasks.append(ms1_scan_params)
#                             self.current_task_id += 1
#                         fragmented_count += 1
#                 else:
#                     # create a new ms2 scan parameter to be sent to the mass spec
#                     precursor_scan_id = self.scan_to_process.scan_id
#                     dda_scan_params = self.get_ms2_scan_params(mz, intensity, precursor_scan_id,
#                                                                current_isolation_width, current_mz_tol, current_rt_tol)
#                     self.current_task_id += 1
#                     new_tasks.append(dda_scan_params)
#                     fragmented_count += 1
#
#             # an MS1 is added here, as we no longer send MS1s as default
#             ms1_scan_params = self.get_ms1_scan_params()
#             new_tasks.append(ms1_scan_params)
#             self.current_task_id += 1
#             self.next_processed_scan_id = self.current_task_id
#
#             # create temp exclusion items
#             self.temp_exclusion_list = self._update_temp_exclusion_list(ms2_tasks)
#
#             # set this ms1 scan as has been processed
#             self.scan_to_process = None
#         return new_tasks
#
#     def _get_current_N_DEW(self, time):
#         idx = np.nonzero(self.scan_param_changepoints <= time)[0][-1]
#         current_N = self.N[idx]
#         current_rt_tol = self.rt_tols[idx]
#         return current_N, current_rt_tol, idx

# Some potentially useful skeleton code for simplifying controllers

# def _process_scan(self, scan):
#     new_tasks = []  # this gets updated in _get_ms2_scan
#     fragmented_count = 0  # this gets updated in _get_ms2_scan
#     if self.scan_to_process is not None:
#         self._update_status()  # updates ROIs, exclusion list etc
#         self.get_precursor_info()  # retrieves relevant precursor info (mz, int, rt) etc.
#         # Above should be consistent for ROI methods and consistent for
#         scores_info = self.get_scores()  # this is the individual scoring system
#         for i in scores_info:
#             new_task, fragmented_count = self.get_ms2_scan(scan, fragmented_count)  # gets ms2 scan, returns None if ms2 scan not chosen
#             # in above we update the fragmentation_count and things like N
#             new_tasks.append(new_task)
#             self.update_exclusion()  # updates exlcusion list
#             self.update_scores()  # updated scores. Generally does nothing, people useful for some future methods
#             if new_task == None:
#                 pass  # stop looping through scans
#         return new_tasks
