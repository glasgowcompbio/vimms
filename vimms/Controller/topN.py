import bisect
from collections import defaultdict

import numpy as np
from loguru import logger
from mass_spec_utils.data_import.mzmine import load_picked_boxes

from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION, DEFAULT_MS1_ACTIVATION_TYPE, DEFAULT_MS1_MASS_ANALYSER, DEFAULT_MS1_ISOLATION_MODE, \
    DEFAULT_MS2_ACTIVATION_TYPE, DEFAULT_MS2_MASS_ANALYSER, DEFAULT_MS2_ISOLATION_MODE
from vimms.Controller.base import Controller
from vimms.MassSpec import ScanParameters, ExclusionItem


class TopNController(Controller):
    """
    A Top-N controller. Does an MS1 scan followed by N fragmentation scans of the peaks with the highest intensity
    that are not excluded
    """

    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE):
        super().__init__()
        self.ionisation_mode = ionisation_mode
        self.N = N
        self.isolation_width = isolation_width  # the isolation width (in Dalton) to select a precursor ion
        self.mz_tol = mz_tol  # the m/z window (ppm) to prevent the same precursor ion to be fragmented again
        self.rt_tol = rt_tol  # the rt window to prevent the same precursor ion to be fragmented again
        self.min_ms1_intensity = min_ms1_intensity  # minimum ms1 intensity to fragment
        self.ms1_shift = ms1_shift  # number of scans to move ms1 scan forward in list of new_tasks

        # for dynamic exclusion window
        self.exclusion_list = []  # a list of ExclusionItem
        self.temp_exclusion_list = []

        # advanced parameters
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution
        self.ms1_activation_type = ms1_activation_type
        self.ms1_mass_analyser = ms1_mass_analyser
        self.ms1_isolation_mode = ms1_isolation_mode

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution
        self.ms2_activation_type = ms2_activation_type
        self.ms2_mass_analyser = ms2_mass_analyser
        self.ms2_isolation_mode = ms2_isolation_mode

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        fragmented_count = 0
        if self.scan_to_process is not None:
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt

            # loop over points in decreasing intensity
            idx = np.argsort(intensities)[::-1]

            done_ms1 = False
            ms2_tasks = []
            for i in idx:
                mz = mzs[i]
                intensity = intensities[i]

                # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
                if fragmented_count >= self.N:
                    logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                    break

                if intensity < self.min_ms1_intensity:
                    logger.debug(
                        'Time %f Minimum intensity threshold %f reached at %f, %d' % (
                            rt, self.min_ms1_intensity, intensity, fragmented_count))
                    break

                # skip ion in the dynamic exclusion list of the mass spec
                if self._is_excluded(mz, rt):
                    continue

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                      self.isolation_width, self.mz_tol, self.rt_tol,
                                                                      agc_target=self.ms2_agc_target,
                                                                      max_it=self.ms2_max_it,
                                                                      collision_energy=self.ms2_collision_energy,
                                                                      orbitrap_resolution=self.ms2_orbitrap_resolution,
                                                                      activation_type=self.ms2_activation_type,
                                                                      mass_analyser=self.ms2_mass_analyser,
                                                                      isolation_mode=self.ms2_isolation_mode)
                new_tasks.append(dda_scan_params)
                ms2_tasks.append(dda_scan_params)
                fragmented_count += 1
                self.current_task_id += 1

                # add an ms1 here
                if fragmented_count == self.N - self.ms1_shift:
                    ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                               max_it=self.ms1_max_it,
                                                                               collision_energy=self.ms1_collision_energy,
                                                                               orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                               activation_type=self.ms1_activation_type,
                                                                               mass_analyser=self.ms1_mass_analyser,
                                                                               isolation_mode=self.ms1_isolation_mode)
                    self.current_task_id += 1
                    self.next_processed_scan_id = self.current_task_id
                    new_tasks.append(ms1_scan_params)
                    done_ms1 = True

            # if no ms1 has been added, then add at the end
            if not done_ms1:
                # if fragmented_count < self.N - self.ms1_shift:
                ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                           max_it=self.ms1_max_it,
                                                                           collision_energy=self.ms1_collision_energy,
                                                                           orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                           activation_type=self.ms1_activation_type,
                                                                           mass_analyser=self.ms1_mass_analyser,
                                                                           isolation_mode=self.ms1_isolation_mode)
                self.current_task_id += 1
                self.next_processed_scan_id = self.current_task_id
                new_tasks.append(ms1_scan_params)

            # create temp exclusion items
            # tasks = new_tasks[min(self.N - self.ms1_shift+1, len(new_tasks)):max(self.N - self.ms1_shift+1, len(new_tasks))]
            # self.temp_exclusion_list = self._update_temp_exclusion_list(tasks)
            self.temp_exclusion_list = self._update_temp_exclusion_list(ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def _update_temp_exclusion_list(self, tasks):
        temp_exclusion_list = []
        rt = self.scan_to_process.rt
        for task in tasks:
            mz = task.get('precursor_mz').precursor_mz
            mz_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL)
            rt_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
            mz_lower = mz * (1 - mz_tol / 1e6)
            mz_upper = mz * (1 + mz_tol / 1e6)
            rt_lower = rt
            rt_upper = rt + rt_tol
            x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper)
            logger.debug('Time {:.6f} Created dynamic temporary exclusion window mz ({}-{}) rt ({}-{})'.format(
                rt,
                x.from_mz, x.to_mz, x.from_rt, x.to_rt
            ))
            temp_exclusion_list.append(x)
        return temp_exclusion_list

    def update_state_after_scan(self, last_scan):
        # the DEW list update must be done after time has been increased
        self._manage_dynamic_exclusion_list(last_scan)

    def reset(self):
        self.exclusion_list = []

    def _manage_dynamic_exclusion_list(self, scan):
        """
        Manages dynamic exclusion list
        :param param: a scan parameter object
        :param scan: the newly generated scan
        :return: None
        """
        # FIXME: maybe not correct, see the step() method in IAPIMassSpec class
        if scan.scan_duration is not None:
            current_time = scan.rt + scan.scan_duration
        else:
            current_time = scan.rt

        if scan.ms_level >= 2:  # if ms-level is 2, it's a custom scan and we should always know its scan parameters
            assert scan.scan_params is not None
            precursor = scan.scan_params.get(ScanParameters.PRECURSOR_MZ)

            # add dynamic exclusion item to the exclusion list to prevent the same precursor ion being fragmented
            # multiple times in the same mz and rt window
            # Note: at this point, fragmentation has occurred and time has been incremented! so the time when
            # items are checked for dynamic exclusion is the time when MS2 fragmentation occurs
            # TODO: we need to add a repeat count too, i.e. how many times we've seen a fragment peak before
            #  it gets excluded (now it's basically 1)

            # TODO: check if already excluded and, if so, just move the time
            mz = precursor.precursor_mz
            mz_tol = scan.scan_params.get(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL)
            rt_tol = scan.scan_params.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
            mz_lower = mz * (1 - mz_tol / 1e6)
            mz_upper = mz * (1 + mz_tol / 1e6)
            rt_lower = current_time
            rt_upper = current_time + rt_tol
            x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper)
            logger.debug('Time {:.6f} Created dynamic exclusion window mz ({}-{}) rt ({}-{})'.format(
                current_time,
                x.from_mz, x.to_mz, x.from_rt, x.to_rt
            ))
            self.exclusion_list.append(x)

        # remove expired items from dynamic exclusion list
        self.exclusion_list = list(filter(lambda x: x.to_rt > current_time, self.exclusion_list))

    def _is_excluded(self, mz, rt):
        """
        Checks if a pair of (mz, rt) value is currently excluded by dynamic exclusion window
        :param mz: m/z value
        :param rt: RT value
        :return: True if excluded, False otherwise
        """
        # TODO: make this faster?
        exclusion_list = self.exclusion_list + self.temp_exclusion_list
        for x in exclusion_list:
            exclude_mz = x.from_mz <= mz <= x.to_mz
            exclude_rt = x.from_rt <= rt <= x.to_rt
            if exclude_mz and exclude_rt:
                logger.debug(
                    'Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(mz, rt, x))
                return True
        return False


class ScanItem(object):
    """
    Represents a scan item object. Used by the WeightedDEW controller.
    """

    def __init__(self, mz, intensity, weight=1):
        self.mz = mz
        self.intensity = intensity
        self.weight = weight

    def __lt__(self, other):
        if self.intensity * self.weight <= other.intensity * other.weight:
            return True
        else:
            return False


class WeightedDEWController(TopNController):
    """
    A Top-N controller. Does an MS1 scan followed by N fragmentation scans of the peaks with the highest intensity
    that are not excluded
    """

    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift=0,
                 exclusion_t_0=15, log_intensity=False,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE):
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, ms1_shift=ms1_shift,
                         ms1_agc_target=ms1_agc_target,
                         ms1_max_it=ms1_max_it,
                         ms1_collision_energy=ms1_collision_energy,
                         ms1_orbitrap_resolution=ms1_orbitrap_resolution,
                         ms1_activation_type=ms1_activation_type,
                         ms1_mass_analyser=ms1_mass_analyser,
                         ms1_isolation_mode=ms1_isolation_mode,
                         ms2_agc_target=ms2_agc_target,
                         ms2_max_it=ms2_max_it,
                         ms2_collision_energy=ms2_collision_energy,
                         ms2_orbitrap_resolution=ms2_orbitrap_resolution,
                         ms2_activation_type=ms2_activation_type,
                         ms2_mass_analyser=ms2_mass_analyser,
                         ms2_isolation_mode=ms2_isolation_mode)
        self.exclusion_t_0 = exclusion_t_0
        self.log_intensity = log_intensity
        assert self.exclusion_t_0 <= self.rt_tol

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        fragmented_count = 0
        if self.scan_to_process is not None:
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt

            if not self.log_intensity:
                mzi = [ScanItem(mz, intensities[i]) for i, mz in enumerate(mzs) if
                       intensities[i] >= self.min_ms1_intensity]
            else:
                # take log of intensities for peak scoring
                mzi = [ScanItem(mz, np.log(intensities[i])) for i, mz in enumerate(mzs) if
                       intensities[i] >= self.min_ms1_intensity]

            for si in mzi:
                is_exc, weight = self._is_excluded(si.mz, rt)
                si.weight = weight

            mzi.sort(reverse=True)

            done_ms1 = False
            ms2_tasks = []
            for i in range(len(mzi)):
                # mz = mzi[i].mz
                # intensity = mzi[i].intensity
                # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
                if fragmented_count >= self.N:
                    logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                    break

                mz = mzi[i].mz
                if not self.log_intensity:
                    intensity = mzi[i].intensity
                else:
                    intensity = np.exp(mzi[i].intensity)

                # if 138 <= mz <= 138.5:
                #     print(mz,intensity,mzi[i].weight)

                if mzi[i].weight == 0.0:
                    logger.debug(
                        'Time %f no ions left reached at %f, %d' % (
                            rt, intensity, fragmented_count))
                    break

                # create a new ms2 scan parameter to be sent to the mass spec
                precursor_scan_id = self.scan_to_process.scan_id
                dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                      self.isolation_width, self.mz_tol, self.rt_tol,
                                                                      agc_target=self.ms2_agc_target,
                                                                      max_it=self.ms2_max_it,
                                                                      collision_energy=self.ms2_collision_energy,
                                                                      orbitrap_resolution=self.ms2_orbitrap_resolution,
                                                                      activation_type=self.ms2_activation_type,
                                                                      mass_analyser=self.ms2_mass_analyser,
                                                                      isolation_mode=self.ms2_isolation_mode)
                new_tasks.append(dda_scan_params)
                ms2_tasks.append(dda_scan_params)
                fragmented_count += 1
                self.current_task_id += 1

                # add an ms1 here
                if fragmented_count == self.N - self.ms1_shift:
                    ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                               max_it=self.ms1_max_it,
                                                                               collision_energy=self.ms1_collision_energy,
                                                                               orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                               activation_type=self.ms1_activation_type,
                                                                               mass_analyser=self.ms1_mass_analyser,
                                                                               isolation_mode=self.ms1_isolation_mode)
                    self.current_task_id += 1
                    self.next_processed_scan_id = self.current_task_id
                    new_tasks.append(ms1_scan_params)
                    done_ms1 = True

            # if no ms1 has been added, then add at the end
            if not done_ms1:
                # if fragmented_count < self.N - self.ms1_shift:
                ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                           max_it=self.ms1_max_it,
                                                                           collision_energy=self.ms1_collision_energy,
                                                                           orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                           activation_type=self.ms1_activation_type,
                                                                           mass_analyser=self.ms1_mass_analyser,
                                                                           isolation_mode=self.ms1_isolation_mode)
                self.current_task_id += 1
                self.next_processed_scan_id = self.current_task_id
                new_tasks.append(ms1_scan_params)

            # create temp exclusion items
            # tasks = new_tasks[min(self.N - self.ms1_shift+1, len(new_tasks)):max(self.N - self.ms1_shift+1, len(new_tasks))]
            # self.temp_exclusion_list = self._update_temp_exclusion_list(tasks)
            self.temp_exclusion_list = self._update_temp_exclusion_list(ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def _manage_dynamic_exclusion_list(self, scan):
        # self.exclusion_list = set(self.exclusion_list)
        # for ei in self.remove_exclusion_items:
        #     if ei in self.exclusion_list:
        #         self.exclusion_list.remove(ei)
        # self.exclusion_list = list(self.exclusion_list)
        super()._manage_dynamic_exclusion_list(scan)

    def _is_excluded(self, mz, rt):
        """
        Checks if a pair of (mz, rt) value is currently excluded by dynamic exclusion window
        :param mz: m/z value
        :param rt: RT value
        :return: True if excluded, False otherwise
        """
        # TODO: make this faster?
        exclusion_list = self.exclusion_list + self.temp_exclusion_list
        exclusion_list.sort(key=lambda x: x.from_rt, reverse=True)
        for x in exclusion_list:
            exclude_mz = x.from_mz <= mz <= x.to_mz
            exclude_rt = x.from_rt <= rt <= x.to_rt
            if exclude_mz and exclude_rt:
                logger.debug(
                    'Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(mz, rt, x))
                if rt <= x.from_rt + self.exclusion_t_0:
                    return True, 0.0
                else:
                    weight = (rt - (self.exclusion_t_0 + x.from_rt)) / (self.rt_tol - self.exclusion_t_0)
                    assert weight <= 1, weight
                    # self.remove_exclusion_items.append(x)
                    return True, weight
        return False, 1


class OptimalTopNController(TopNController):
    def __init__(self, ionisation_mode, N,
                 isolation_widths, mz_tols, rt_tols, min_ms1_intensity, box_file, score_method='intensity',
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE):
        super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_shift=0,
                         ms1_agc_target=ms1_agc_target,
                         ms1_max_it=ms1_max_it,
                         ms1_collision_energy=ms1_collision_energy,
                         ms1_orbitrap_resolution=ms1_orbitrap_resolution,
                         ms1_activation_type=ms1_activation_type,
                         ms1_mass_analyser=ms1_mass_analyser,
                         ms1_isolation_mode=ms1_isolation_mode,
                         ms2_agc_target=ms2_agc_target,
                         ms2_max_it=ms2_max_it,
                         ms2_collision_energy=ms2_collision_energy,
                         ms2_orbitrap_resolution=ms2_orbitrap_resolution,
                         ms2_activation_type=ms2_activation_type,
                         ms2_mass_analyser=ms2_mass_analyser,
                         ms2_isolation_mode=ms2_isolation_mode)
        if type(box_file) == str:
            self.box_file = box_file
            self._load_boxes()
        else:
            self.boxes = box_file

        self.score_method = score_method

    def _load_boxes(self):
        self.boxes = load_picked_boxes(self.box_file)
        logger.debug("Loaded {} boxes".format(len(self.boxes)))

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        ms2_tasks = []
        if self.scan_to_process is not None:

            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt

            # Find boxes that span the current rt value
            sub_boxes = list(
                filter(lambda x: x.rt_range_in_seconds[0] <= rt and x.rt_range_in_seconds[1] >= rt, self.boxes))
            mzi = zip(mzs, intensities)
            # remove any peaks below min intensity
            mzi = list(filter(lambda x: x[1] >= self.min_ms1_intensity, mzi))
            # sort by mz for matching with the boxes
            mzi.sort(key=lambda x: x[0])
            sub_boxes.sort(key=lambda x: x.mz_range[0])
            mzib = self._box_match(mzi, sub_boxes)  # (mz,intensity,box)

            # If there are things to fragment, schedule the scans...
            if len(mzib) > 0:
                # compute the scores
                mzibs = self._score_peak_boxes(mzib, rt, score=self.score_method)
                # loop over points in decreasing score
                fragmented_count = 0
                # idx = np.argsort(intensities)[::-1]
                mzs, intensities, matched_boxes, scores = zip(*mzibs)
                idx = np.argsort(scores)[::-1]

                for i in idx:
                    mz = mzs[i]
                    intensity = intensities[i]
                    matched_box = matched_boxes[i]

                    # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
                    if fragmented_count >= self.N:
                        logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                        break

                    # if intensity < self.min_ms1_intensity:
                    #     logger.debug(
                    #         'Time %f Minimum intensity threshold %f reached at %f, %d' % (
                    #             rt, self.min_ms1_intensity, intensity, fragmented_count))
                    #     break

                    # skip ion in the dynamic exclusion list of the mass spec
                    # if self._is_excluded(mz, rt):
                    #     continue

                    # check if it is one of the picked boxes
                    # sub_sub_boxes = list(filter(lambda x: x.mz_range[0] <= mz and x.mz_range[1] >= mz,sub_boxes))
                    # if len(sub_sub_boxes) == 0:
                    #     # do not fragment
                    #     continue
                    # else:
                    #     # remove this box so it isn't fragmented again
                    #     pos = self.boxes.index(sub_sub_boxes[0])
                    #     del self.boxes[pos]
                    #     pos2 = sub_boxes.index(sub_sub_boxes[0])
                    #     del sub_boxes[pos2]

                    # create a new ms2 scan parameter to be sent to the mass spec
                    precursor_scan_id = self.scan_to_process.scan_id
                    dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                          self.isolation_width, self.mz_tol,
                                                                          self.rt_tol,
                                                                          agc_target=self.ms2_agc_target,
                                                                          max_it=self.ms2_max_it,
                                                                          collision_energy=self.ms2_collision_energy,
                                                                          orbitrap_resolution=self.ms2_orbitrap_resolution,
                                                                          activation_type=self.ms2_activation_type,
                                                                          mass_analyser=self.ms2_mass_analyser,
                                                                          isolation_mode=self.ms2_isolation_mode)
                    new_tasks.append(dda_scan_params)
                    ms2_tasks.append(dda_scan_params)
                    fragmented_count += 1
                    self.current_task_id += 1

                    pos = self.boxes.index(matched_box)
                    del self.boxes[pos]

            # an MS1 is added here, as we no longer send MS1s as default
            ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                       max_it=self.ms1_max_it,
                                                                       collision_energy=self.ms1_collision_energy,
                                                                       orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                       activation_type=self.ms1_activation_type,
                                                                       mass_analyser=self.ms1_mass_analyser,
                                                                       isolation_mode=self.ms1_isolation_mode)
            self.current_task_id += 1
            self.next_processed_scan_id = self.current_task_id
            new_tasks.append(ms1_scan_params)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def _box_match(self, mzi, boxes):
        # mzi and boxes are sorted by mz
        # loop over the mzi
        mzib = []
        lower_boxes = [b.mz_range[0] for b in boxes]
        for this_mzi in mzi:
            # find the possible boxes
            left_pos = bisect.bisect_right(lower_boxes, this_mzi[0])
            if left_pos < len(boxes):
                left_pos -= 1  # this is the first possible box
                if left_pos == -1:  # peak is lower in mz than all boxes
                    continue
                if this_mzi[0] < boxes[left_pos].mz_range[1]:
                    # found a match
                    # compute time proportion left in the peak
                    matching_box = boxes[left_pos]
                    mzib.append((this_mzi[0], this_mzi[1], matching_box))
                    del boxes[left_pos]
                    del lower_boxes[left_pos]
                else:
                    # no match found
                    pass
            else:
                # no match found
                pass
        return mzib

    def _score_peak_boxes(self, mzib, current_rt, score='intensity'):
        # mzib = (mz,intensity,box) tuple
        if score == 'intensity':
            # simplest: score = intensity
            scores = [(mz, i, b, i) for mz, i, b in mzib]
        elif score == 'urgency':
            scores = [(mz, i, b, -(b.rt_range_in_seconds[1] - current_rt)) for mz, i, b in mzib]
        elif score == 'apex':
            scores = [(mz, i, b, current_rt - b.rt_in_seconds) for mz, i, b in mzib]
        elif score == 'random':
            scores = [(mz, i, b, np.random.rand()) for mz, i, b in mzib]
        return scores


class PurityController(TopNController):
    def __init__(self, ionisation_mode, N, scan_param_changepoints,
                 isolation_widths, mz_tols, rt_tols, min_ms1_intensity,
                 n_purity_scans=None, purity_shift=None, purity_threshold=0, purity_randomise=True,
                 purity_add_ms1=True, ms1_shift=0,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms1_activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                 ms1_mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
                 ms1_isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                 ms2_activation_type=DEFAULT_MS2_ACTIVATION_TYPE,
                 ms2_mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                 ms2_isolation_mode=DEFAULT_MS2_ISOLATION_MODE):
        super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_shift=0,
                         ms1_agc_target=ms1_agc_target,
                         ms1_max_it=ms1_max_it,
                         ms1_collision_energy=ms1_collision_energy,
                         ms1_orbitrap_resolution=ms1_orbitrap_resolution,
                         ms1_activation_type=ms1_activation_type,
                         ms1_mass_analyser=ms1_mass_analyser,
                         ms1_isolation_mode=ms1_isolation_mode,
                         ms2_agc_target=ms2_agc_target,
                         ms2_max_it=ms2_max_it,
                         ms2_collision_energy=ms2_collision_energy,
                         ms2_orbitrap_resolution=ms2_orbitrap_resolution,
                         ms2_activation_type=ms2_activation_type,
                         ms2_mass_analyser=ms2_mass_analyser,
                         ms2_isolation_mode=ms2_isolation_mode)

        # make sure these are stored as numpy arrays
        self.N = np.array(N)
        self.isolation_width = np.array(isolation_widths)  # the isolation window (in Dalton) to select a precursor ion
        self.mz_tols = np.array(
            mz_tols)  # the m/z window (ppm) to prevent the same precursor ion to be fragmented again
        self.rt_tols = np.array(rt_tols)  # the rt window to prevent the same precursor ion to be fragmented again
        if scan_param_changepoints is not None:
            self.scan_param_changepoints = np.array([0] + scan_param_changepoints)
        else:
            self.scan_param_changepoints = np.array([0])

        # purity stuff
        self.n_purity_scans = n_purity_scans
        self.purity_shift = purity_shift
        self.purity_threshold = purity_threshold
        self.purity_randomise = purity_randomise
        self.purity_add_ms1 = purity_add_ms1

        # make sure the input are all correct
        assert len(self.N) == len(self.scan_param_changepoints) == len(self.isolation_width) == len(
            self.mz_tols) == len(self.rt_tols)
        if self.purity_threshold != 0:
            assert all(self.n_purity_scans <= np.array(self.N))

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.scan_to_process is not None:
            # check queue size because we want to schedule both ms1 and ms2 in the hybrid controller

            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities
            rt = self.scan_to_process.rt

            # set up current scan parameters
            current_N, current_rt_tol, idx = self._get_current_N_DEW(rt)
            current_isolation_width = self.isolation_width[idx]
            current_mz_tol = self.mz_tols[idx]

            # calculate purities
            purities = []
            for mz_idx in range(len(self.scan_to_process.mzs)):
                nearby_mzs_idx = np.where(
                    abs(self.scan_to_process.mzs - self.scan_to_process.mzs[mz_idx]) < current_isolation_width)
                if len(nearby_mzs_idx[0]) == 1:
                    purities.append(1)
                else:
                    total_intensity = sum(self.scan_to_process.intensities[nearby_mzs_idx])
                    purities.append(self.scan_to_process.intensities[mz_idx] / total_intensity)

            # loop over points in decreasing intensity
            fragmented_count = 0
            idx = np.argsort(intensities)[::-1]
            ms2_tasks = []
            for i in idx:
                mz = mzs[i]
                intensity = intensities[i]
                purity = purities[i]

                # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
                if fragmented_count >= current_N:
                    logger.debug('Top-%d ions have been selected' % (current_N))
                    break

                if intensity < self.min_ms1_intensity:
                    logger.debug(
                        'Minimum intensity threshold %f reached at %f, %d' % (
                            self.min_ms1_intensity, intensity, fragmented_count))
                    break

                # skip ion in the dynamic exclusion list of the mass spec
                if self._is_excluded(mz, rt):
                    continue

                if purity <= self.purity_threshold:
                    purity_shift_amounts = [self.purity_shift * (i - (self.n_purity_scans - 1) / 2) for i in
                                            range(self.n_purity_scans)]
                    if self.purity_randomise:
                        purity_randomise_idx = np.random.choice(self.n_purity_scans, self.n_purity_scans, replace=False)
                    else:
                        purity_randomise_idx = range(self.n_purity_scans)
                    for purity_idx in purity_randomise_idx:
                        # create a new ms2 scan parameter to be sent to the mass spec
                        precursor_scan_id = self.scan_to_process.scan_id
                        dda_scan_params = self.environment.get_dda_scan_param(mz + purity_shift_amounts[purity_idx],
                                                                              intensity, precursor_scan_id,
                                                                              current_isolation_width, current_mz_tol,
                                                                              current_rt_tol,
                                                                              agc_target=self.ms2_agc_target,
                                                                              max_it=self.ms2_max_it,
                                                                              collision_energy=self.ms2_collision_energy,
                                                                              orbitrap_resolution=self.ms2_orbitrap_resolution,
                                                                              activation_type=self.ms2_activation_type,
                                                                              mass_analyser=self.ms2_mass_analyser,
                                                                              isolation_mode=self.ms2_isolation_mode)
                        new_tasks.append(dda_scan_params)
                        ms2_tasks.append(dda_scan_params)
                        self.current_task_id += 1
                        if self.purity_add_ms1 and purity_idx != purity_randomise_idx[-1]:
                            ms1_scan_params = self.environment.get_default_scan_params()
                            new_tasks.append(ms1_scan_params)
                            self.current_task_id += 1
                        fragmented_count += 1
                else:
                    # create a new ms2 scan parameter to be sent to the mass spec
                    precursor_scan_id = self.scan_to_process.scan_id
                    dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                          current_isolation_width, current_mz_tol,
                                                                          current_rt_tol,
                                                                          agc_target=self.ms2_agc_target,
                                                                          max_it=self.ms2_max_it,
                                                                          collision_energy=self.ms2_collision_energy,
                                                                          orbitrap_resolution=self.ms2_orbitrap_resolution,
                                                                          activation_type=self.ms2_activation_type,
                                                                          mass_analyser=self.ms2_mass_analyser,
                                                                          isolation_mode=self.ms2_isolation_mode)
                    self.current_task_id += 1
                    new_tasks.append(dda_scan_params)
                    fragmented_count += 1

            # an MS1 is added here, as we no longer send MS1s as default
            ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                       max_it=self.ms1_max_it,
                                                                       collision_energy=self.ms1_collision_energy,
                                                                       orbitrap_resolution=self.ms1_orbitrap_resolution,
                                                                       activation_type=self.ms1_activation_type,
                                                                       mass_analyser=self.ms1_mass_analyser,
                                                                       isolation_mode=self.ms1_isolation_mode)
            new_tasks.append(ms1_scan_params)
            self.current_task_id += 1
            self.next_processed_scan_id = self.current_task_id

            # create temp exclusion items
            self.temp_exclusion_list = self._update_temp_exclusion_list(ms2_tasks)

            # set this ms1 scan as has been processed
            self.scan_to_process = None
        return new_tasks

    def update_state_after_scan(self, last_scan):
        super().update_state_after_scan(last_scan)

    def reset(self):
        super().reset()

    def _get_current_N_DEW(self, time):
        idx = np.nonzero(self.scan_param_changepoints <= time)[0][-1]
        current_N = self.N[idx]
        current_rt_tol = self.rt_tols[idx]
        return current_N, current_rt_tol, idx
