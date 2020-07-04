from collections import defaultdict

import numpy as np
import pandas as pd
import pylab as plt
from loguru import logger

from vimms.Common import DEFAULT_MS1_AGC_TARGET, DEFAULT_MS1_MAXIT, DEFAULT_MS1_COLLISION_ENERGY, \
    DEFAULT_MS1_ORBITRAP_RESOLUTION, DEFAULT_MS2_AGC_TARGET, DEFAULT_MS2_MAXIT, DEFAULT_MS2_COLLISION_ENERGY, \
    DEFAULT_MS2_ORBITRAP_RESOLUTION
from vimms.MassSpec import ScanParameters, ExclusionItem


# from ms2_matching import load_picked_boxes


class Controller(object):
    def __init__(self):
        self.scans = defaultdict(list)  # key: ms level, value: list of scans for that level
        self.make_plot = False
        self.last_ms1_scan = None
        self.environment = None

    def set_environment(self, env):
        self.environment = env

    def handle_scan(self, scan, outgoing_queue_size, pending_tasks_size):
        # record every scan that we've received
        logger.debug('Time %f Received %s' % (scan.rt, scan))
        self.scans[scan.ms_level].append(scan)

        # plot scan if there are peaks
        if scan.num_peaks > 0:
            self._plot_scan(scan)

        # we get an ms1 scan and it has some peaks AND all the pending tasks have been sent and processed AND
        # this ms1 scan is a custom scan we'd sent before (not a method scan)
        # then store it for fragmentation next time
        if scan.ms_level == 1 and \
                scan.num_peaks > 0 and \
                outgoing_queue_size == 0 and \
                pending_tasks_size == 0 and \
                scan.scan_params is not None:
            self.last_ms1_scan = scan
        else:
            self.last_ms1_scan = None

        logger.debug('outgoing_queue_size = %d, pending_tasks_size = %d' % (outgoing_queue_size, pending_tasks_size))
        logger.debug('scan.scan_params = %s' % scan.scan_params)
        logger.debug('last_ms1_scan = %s' % self.last_ms1_scan)

        # impelemnted by subclass
        new_tasks = self._process_scan(scan)
        return new_tasks

    def update_state_after_scan(self, last_scan):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def _process_scan(self, scan):
        raise NotImplementedError()

    def _plot_scan(self, scan):
        if self.make_plot:
            plt.figure()
            for i in range(scan.num_peaks):
                x1 = scan.mzs[i]
                x2 = scan.mzs[i]
                y1 = 0
                y2 = scan.intensities[i]
                a = [[x1, y1], [x2, y2]]
                plt.plot(*zip(*a), marker='', color='r', ls='-', lw=1)
            plt.title('Scan {0} {1}s -- {2} peaks'.format(scan.scan_id, scan.rt, scan.num_peaks))
            plt.show()


class IdleController(Controller):
    """
    A controller that doesn't do any controlling.
    """

    def __init__(self):
        super().__init__()

    def _process_scan(self, scan):
        new_tasks = []
        return new_tasks

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass


class SimpleMs1Controller(Controller):
    """
    A simple MS1 controller which does a full scan of the chemical sample, but no fragmentation
    """

    def __init__(self,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION):
        super().__init__()
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution

    def _process_scan(self, scan):
        task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                        max_it=self.ms1_max_it,
                                                        collision_energy=self.ms1_collision_energy,
                                                        orbitrap_resolution=self.ms1_orbitrap_resolution)
        return [task]

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass


class ScanParameterController(SimpleMs1Controller):
    def __init__(self,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 agc_values = [1e5,2e5,5e5,1e6],
                 maxit_values = [50,100,250,500]):
        super().__init__(ms1_agc_target=ms1_agc_target,
                 ms1_max_it=ms1_max_it,
                 ms1_collision_energy=ms1_collision_energy,
                 ms1_orbitrap_resolution=ms1_orbitrap_resolution)
        self.agc_values = agc_values
        self.maxit_values = maxit_values
        self.scan_number = 0
        self.scan_param_list = []
        # make a list with all combinations of parameters
        for a in self.agc_values:
            for b in self.maxit_values:
                self.scan_param_list.append((a,b))
    
    def _process_scan(self, scan):

        n_scan_params = len(self.scan_param_list)
        agc,maxit = self.scan_param_list[self.scan_number % n_scan_params]

        task = self.environment.get_default_scan_params(agc_target=agc,
                                                        max_it=maxit,
                                                        collision_energy=self.ms1_collision_energy,
                                                        orbitrap_resolution=self.ms1_orbitrap_resolution)
        self.scan_number += 1
        
        return [task]
                

class FixedScheduleController(Controller):
    """
    A controller which takes a schedule of scans, converts them into tasks in queue
    """

    def __init__(self, schedule_file, isolation_width, mz_tol, mass_spec_ionisation_mode, N, rt_tol,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__()
        self.first_scan = True
        self.isolation_width = isolation_width
        self.mz_tol = mz_tol
        self.mass_spec_ionisation_mode = mass_spec_ionisation_mode
        self.schedule = pd.read_csv(schedule_file, header=None, names=["ms_level", "mz", "time", "box_id", "box_file"])
        self.schedule = self.schedule.iloc[range(0, self.schedule.shape[0], 2)]
        self.scheduled_tasks = self._process_schedule()
        self.N = N
        self.rt_tol = rt_tol

        # advanced parameters
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution

    def _process_scan(self, scan):
        if self.first_scan:
            self.first_scan = False
            return self.scheduled_tasks
        else:
            return []

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass

    def _process_schedule(self):
        # converts schedule into tasks and pushes them to the queue
        tasks = []
        for i in range(self.schedule.shape[0]):
            if self.schedule.iloc[i]['ms_level'] == 'MS1':
                new_task = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                    max_it=self.ms1_max_it,
                                                                    collision_energy=self.ms1_collision_energy,
                                                                    orbitrap_resolution=self.ms1_orbitrap_resolution)
            else:
                precursor_scan_id = self.last_ms1_scan.scan_id
                mz = self.schedule.iloc[i]['mz']
                intensity = 0
                rt_tol = 0
                new_task = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                               self.isolation_width, self.mz_tol, rt_tol,
                                                               agc_target=self.ms2_agc_target,
                                                               max_it=self.ms2_max_it,
                                                               collision_energy=self.ms2_collision_energy,
                                                               orbitrap_resolution=self.ms2_orbitrap_resolution)
            tasks.append(new_task)
        return tasks


class TopNController(Controller):
    """
    A Top-N controller. Does an MS1 scan followed by N fragmentation scans of the peaks with the highest intensity
    that are not excluded
    """

    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__()
        self.ionisation_mode = ionisation_mode
        self.N = N
        self.isolation_width = isolation_width  # the isolation width (in Dalton) to select a precursor ion
        self.mz_tol = mz_tol  # the m/z window (ppm) to prevent the same precursor ion to be fragmented again
        self.rt_tol = rt_tol  # the rt window to prevent the same precursor ion to be fragmented again
        self.min_ms1_intensity = min_ms1_intensity  # minimum ms1 intensity to fragment

        # for dynamic exclusion window
        self.exclusion_list = []  # a list of ExclusionItem

        # stores the mapping between precursor peak to ms2 scans
        self.precursor_information = defaultdict(list)  # key: Precursor object, value: ms2 scans

        # advanced parameters
        self.ms1_agc_target = ms1_agc_target
        self.ms1_max_it = ms1_max_it
        self.ms1_collision_energy = ms1_collision_energy
        self.ms1_orbitrap_resolution = ms1_orbitrap_resolution

        self.ms2_agc_target = ms2_agc_target
        self.ms2_max_it = ms2_max_it
        self.ms2_collision_energy = ms2_collision_energy
        self.ms2_orbitrap_resolution = ms2_orbitrap_resolution

    def _process_scan(self, scan):
        # if there's a previous ms1 scan to process
        new_tasks = []
        if self.last_ms1_scan is not None:

            mzs = self.last_ms1_scan.mzs
            intensities = self.last_ms1_scan.intensities
            rt = self.last_ms1_scan.rt

            # loop over points in decreasing intensity
            fragmented_count = 0
            idx = np.argsort(intensities)[::-1]
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
                precursor_scan_id = self.last_ms1_scan.scan_id
                dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                               self.isolation_width, self.mz_tol, self.rt_tol,
                                                               agc_target=self.ms2_agc_target,
                                                               max_it=self.ms2_max_it,
                                                               collision_energy=self.ms2_collision_energy,
                                                               orbitrap_resolution=self.ms2_orbitrap_resolution)
                new_tasks.append(dda_scan_params)
                fragmented_count += 1

            # an MS1 is added here, as we no longer send MS1s as default
            ms1_scan_params = self.environment.get_default_scan_params(agc_target=self.ms1_agc_target,
                                                                max_it=self.ms1_max_it,
                                                                collision_energy=self.ms1_collision_energy,
                                                                orbitrap_resolution=self.ms1_orbitrap_resolution)
            new_tasks.append(ms1_scan_params)

            # set this ms1 scan as has been processed
            self.last_ms1_scan = None
        return new_tasks

    def update_state_after_scan(self, last_scan):
        # add precursor and DEW information based on the current scan produced
        # the DEW list update must be done after time has been increased
        self._add_precursor_info(last_scan)
        self._manage_dynamic_exclusion_list(last_scan)

    def reset(self):
        self.exclusion_list = []
        self.precursor_information = defaultdict(list)

    def _add_precursor_info(self, scan):
        """
            Adds precursor ion information.
            If MS2 and above, and controller tells us which precursor ion the scan is coming from, store it
        :param param: a scan parameter object
        :param scan: the newly generated scan
        :return: None
        """
        if scan.ms_level >= 2:  # if ms-level is 2, it's a custom scan and we should always know its scan parameters
            assert scan.scan_params is not None
            precursor = scan.scan_params.get(ScanParameters.PRECURSOR_MZ)
            isolation_windows = scan.scan_params.compute_isolation_windows()
            iso_min = isolation_windows[0][0][0]  # half-width isolation window, in Da
            iso_max = isolation_windows[0][0][1]  # half-width isolation window, in Da
            logger.debug('Time {:.6f} Isolated precursor ion {:.4f} at ({:.4f}, {:.4f})'.format(scan.rt,
                                                                                                precursor.precursor_mz,
                                                                                                iso_min,
                                                                                                iso_max))
            self.precursor_information[precursor].append(scan)

    def _manage_dynamic_exclusion_list(self, scan):
        """
        Manages dynamic exclusion list
        :param param: a scan parameter object
        :param scan: the newly generated scan
        :return: None
        """
        # FIXME: maybe not correct
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
        for x in self.exclusion_list:
            exclude_mz = x.from_mz <= mz <= x.to_mz
            exclude_rt = x.from_rt <= rt <= x.to_rt
            if exclude_mz and exclude_rt:
                logger.debug(
                    'Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(mz, rt, x))
                return True
        return False


def box_match(mzi, boxes):
    # mzi and boxes are sorted by mz
    import bisect
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


def score_peak_boxes(mzib, current_rt, score='intensity'):
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


class OptimalTopNController(TopNController):
    def __init__(self, ionisation_mode, N,
                 isolation_widths, mz_tols, rt_tols, min_ms1_intensity, box_file, score_method='intensity',
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION):
        super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)
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
        if self.last_ms1_scan is not None:

            mzs = self.last_ms1_scan.mzs
            intensities = self.last_ms1_scan.intensities
            rt = self.last_ms1_scan.rt

            # Find boxes that span the current rt value
            sub_boxes = list(
                filter(lambda x: x.rt_range_in_seconds[0] <= rt and x.rt_range_in_seconds[1] >= rt, self.boxes))
            mzi = zip(mzs, intensities)
            # remove any peaks below min intensity
            mzi = list(filter(lambda x: x[1] >= self.min_ms1_intensity, mzi))
            # sort by mz for matching with the boxes
            mzi.sort(key=lambda x: x[0])
            sub_boxes.sort(key=lambda x: x.mz_range[0])
            mzib = box_match(mzi, sub_boxes)  # (mz,intensity,box)

            # If there are things to fragment, schedule the scans...
            if len(mzib) > 0:
                # compute the scores
                mzibs = score_peak_boxes(mzib, rt, score=self.score_method)
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
                    precursor_scan_id = self.last_ms1_scan.scan_id
                    dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                          self.isolation_width, self.mz_tol,
                                                                          self.rt_tol)
                    new_tasks.append(dda_scan_params)
                    fragmented_count += 1

                    pos = self.boxes.index(matched_box)
                    del self.boxes[pos]

            # an MS1 is added here, as we no longer send MS1s as default
            ms1_scan_params = self.environment.get_default_scan_params()
            new_tasks.append(ms1_scan_params)

            # set this ms1 scan as has been processed
            self.last_ms1_scan = None
        return new_tasks


class PurityController(TopNController):
    def __init__(self, ionisation_mode, N, scan_param_changepoints,
                 isolation_widths, mz_tols, rt_tols, min_ms1_intensity,
                 n_purity_scans=None, purity_shift=None, purity_threshold=0, purity_randomise=True,
                 purity_add_ms1=True,
                 # advanced parameters
                 ms1_agc_target=DEFAULT_MS1_AGC_TARGET,
                 ms1_max_it=DEFAULT_MS1_MAXIT,
                 ms1_collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
                 ms1_orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
                 ms2_agc_target=DEFAULT_MS2_AGC_TARGET,
                 ms2_max_it=DEFAULT_MS2_MAXIT,
                 ms2_collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                 ms2_orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION
                 ):
        super().__init__(ionisation_mode, N, isolation_widths, mz_tols, rt_tols, min_ms1_intensity, ms1_agc_target,
                         ms1_max_it, ms1_collision_energy, ms1_orbitrap_resolution, ms2_agc_target, ms2_max_it,
                         ms2_collision_energy, ms2_orbitrap_resolution)

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
        if self.last_ms1_scan is not None:
            # check queue size because we want to schedule both ms1 and ms2 in the hybrid controller

            mzs = self.last_ms1_scan.mzs
            intensities = self.last_ms1_scan.intensities
            rt = self.last_ms1_scan.rt

            # set up current scan parameters
            current_N, current_rt_tol, idx = self._get_current_N_DEW(rt)
            current_isolation_width = self.isolation_width[idx]
            current_mz_tol = self.mz_tols[idx]

            # calculate purities
            purities = []
            for mz_idx in range(len(self.last_ms1_scan.mzs)):
                nearby_mzs_idx = np.where(
                    abs(self.last_ms1_scan.mzs - self.last_ms1_scan.mzs[mz_idx]) < current_isolation_width)
                if len(nearby_mzs_idx[0]) == 1:
                    purities.append(1)
                else:
                    total_intensity = sum(self.last_ms1_scan.intensities[nearby_mzs_idx])
                    purities.append(self.last_ms1_scan.intensities[mz_idx] / total_intensity)

            # loop over points in decreasing intensity
            fragmented_count = 0
            idx = np.argsort(intensities)[::-1]
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
                        precursor_scan_id = self.last_ms1_scan.scan_id
                        dda_scan_params = self.environment.get_dda_scan_param(mz + purity_shift_amounts[purity_idx],
                                                                              intensity, precursor_scan_id,
                                                                              current_isolation_width, current_mz_tol,
                                                                              current_rt_tol)
                        new_tasks.append(dda_scan_params)
                        if self.purity_add_ms1 and purity_idx != purity_randomise_idx[-1]:
                            ms1_scan_params = self.environment.get_default_scan_params()
                            new_tasks.append(ms1_scan_params)
                        fragmented_count += 1
                else:
                    # create a new ms2 scan parameter to be sent to the mass spec
                    precursor_scan_id = self.last_ms1_scan.scan_id
                    dda_scan_params = self.environment.get_dda_scan_param(mz, intensity, precursor_scan_id,
                                                                          current_isolation_width, current_mz_tol,
                                                                          current_rt_tol)
                    new_tasks.append(dda_scan_params)
                    fragmented_count += 1

            # an MS1 is added here, as we no longer send MS1s as default
            ms1_scan_params = self.environment.get_default_scan_params()
            new_tasks.append(ms1_scan_params)

            # set this ms1 scan as has been processed
            self.last_ms1_scan = None
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


