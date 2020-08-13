import bisect
import math
from collections import OrderedDict
import pandas as pd

import numpy as np
import pylab as plt
import pymzml
from loguru import logger
from scipy.stats import pearsonr

from vimms.Chemicals import ChemicalCreator, UnknownChemical
from vimms.Chromatograms import EmpiricalChromatogram
from vimms.Common import PROTON_MASS, CHEM_NOISE, GET_MS2_BY_PEAKS

POS_TRANSFORMATIONS = OrderedDict()
POS_TRANSFORMATIONS['M+H'] = lambda mz: (mz + PROTON_MASS)
POS_TRANSFORMATIONS['[M+ACN]+H'] = lambda mz: (mz + 42.033823)
POS_TRANSFORMATIONS['[M+CH3OH]+H'] = lambda mz: (mz + 33.033489)
POS_TRANSFORMATIONS['[M+NH3]+H'] = lambda mz: (mz + 18.033823)
POS_TRANSFORMATIONS['M+Na'] = lambda mz: (mz + 22.989218)
POS_TRANSFORMATIONS['M+K'] = lambda mz: (mz + 38.963158)
POS_TRANSFORMATIONS['M+2Na-H'] = lambda mz: (mz + 44.971160)
POS_TRANSFORMATIONS['M+ACN+Na'] = lambda mz: (mz + 64.015765)
POS_TRANSFORMATIONS['M+2Na-H'] = lambda mz: (mz + 44.971160)
POS_TRANSFORMATIONS['M+2K+H'] = lambda mz: (mz + 76.919040)
POS_TRANSFORMATIONS['[M+DMSO]+H'] = lambda mz: (mz + 79.02122)
POS_TRANSFORMATIONS['[M+2ACN]+H'] = lambda mz: (mz + 83.060370)
POS_TRANSFORMATIONS['2M+H'] = lambda mz: (mz * 2) + 1.007276
POS_TRANSFORMATIONS['M+ACN+Na'] = lambda mz: (mz + 64.015765)
POS_TRANSFORMATIONS['2M+NH4'] = lambda mz: (mz * 2) + 18.033823


# Object to store a RoI
# Maintains 3 lists -- mz, rt and intensity
# When a new point (mz,rt,intensity) is added, it updates the 
# list and the mean mz which is required.
class Roi(object):
    def __init__(self, mz, rt, intensity):
        if type(mz) == list:
            self.mz_list = mz
        else:
            self.mz_list = [mz]
        if type(rt) == list:
            self.rt_list = rt
        else:
            self.rt_list = [rt]
        if type(intensity) == list:
            self.intensity_list = intensity
        else:
            self.intensity_list = [intensity]
        self.n = len(self.mz_list)
        self.mz_sum = sum(self.mz_list)
        self.length_in_seconds = self.rt_list[-1] - self.rt_list[0]

    def get_mean_mz(self):
        return self.mz_sum / self.n

    def get_max_intensity(self):
        return max(self.intensity_list)

    def get_min_intensity(self):
        return min(self.intensity_list)

    def get_autocorrelation(self, lag=1):
        return pd.Series(self.intensity_list).autocorr(lag=lag)

    def add(self, mz, rt, intensity):
        self.mz_list.append(mz)
        self.rt_list.append(rt)
        self.intensity_list.append(intensity)
        self.mz_sum += mz
        self.n += 1
        self.length_in_seconds = self.rt_list[-1] - self.rt_list[0]

    def __lt__(self, other):
        return self.get_mean_mz() <= other.get_mean_mz()

    def to_chromatogram(self):
        if self.n == 0:
            return None
        chrom = EmpiricalChromatogram(np.array(self.rt_list), np.array(self.mz_list), np.array(self.intensity_list))
        return chrom

    def __repr__(self):
        return 'ROI with data points=%d mz (%.4f-%.4f) rt (%.4f-%.4f)' % (
            self.n,
            self.mz_list[0], self.mz_list[-1],
            self.rt_list[0], self.rt_list[-1])


INITIAL_WAITING = 0
CAN_FRAGMENT = 1
AFTER_FRAGMENT = 2
POST_PEAK = 3


class SmartRoi(Roi):
    def __init__(self, mz, rt, intensity, initial_length_seconds=5, reset_length_seconds=100,
                 intensity_increase_factor=2, dew=15, drop_perc = 0.01):
        super().__init__(mz, rt, intensity)

        if initial_length_seconds > 0:
            self.status = INITIAL_WAITING
            self.set_can_fragment(False)
        else:
            self.status = CAN_FRAGMENT
            self.set_can_fragment(True)

        self.min_frag_intensity = None

        self.initial_length_seconds = initial_length_seconds
        self.reset_length_seconds = reset_length_seconds
        self.intensity_increase_factor = intensity_increase_factor
        self.drop_perc = drop_perc
        self.dew = dew

    def fragmented(self):
        self.is_fragmented = True
        self.set_can_fragment(False)
        self.fragmented_index = len(self.mz_list)-1
        self.status = AFTER_FRAGMENT

    def get_status(self):
        if self.status == 0:
            return "INITIAL_WAITING"
        elif self.status == 1:
            return "CAN_FRAGMENT"
        elif self.status == 2:
            return "AFTER_FRAGMENT"
        elif self.status == 3:
            return "POST_PEAK"

    def add(self, mz, rt, intensity):
        super().add(mz, rt, intensity)
        if self.status == INITIAL_WAITING:
            if self.length_in_seconds >= self.initial_length_seconds:
                self.status = CAN_FRAGMENT
                self.set_can_fragment(True)
        elif self.status == AFTER_FRAGMENT:  
            # in a period after a fragmentation has happened
            # if enough time has elapsed, reset everything
            if self.rt_list[-1] - self.rt_list[self.fragmented_index] > self.reset_length_seconds:
                self.status = CAN_FRAGMENT
                self.set_can_fragment(True)
            elif self.rt_list[-1] - self.rt_list[self.fragmented_index] > self.dew:
                # standard DEW has expired
                # find the min intensity since the frag
                # check current intensity -- if it is 5* when we fragmented, we can go again
                min_since_frag = min(self.intensity_list[self.fragmented_index:])
                if self.intensity_list[-1] > min_since_frag*self.intensity_increase_factor:
                    self.status = CAN_FRAGMENT
                    self.set_can_fragment(True)
                elif self.intensity_list[-1] < self.drop_perc*self.intensity_list[self.fragmented_index]:
                    # signal has dropped, but ROI still exists.
                    self.status = CAN_FRAGMENT
                    self.set_can_fragment(True)
                    # self.min_frag_intensity = self.intensity_list[-1]*self.intensity_increase_factor
                    
        # code below never happens
        elif self.status == POST_PEAK:
            if self.rt_list[-1] - self.rt_list[self.fragmented_index] > self.dew:
                if self.intensity_list[-1] > self.min_frag_intensity:
                    self.status = CAN_FRAGMENT
                    self.set_can_fragment(True)


    def get_can_fragment(self):
        return self.can_fragment

    def set_can_fragment(self, status):
        self.can_fragment = status

    def get_last_datum(self):
        return (self.mz_list[-1],self.rt_list[-1],self.intensity_list[-1])

# Find the RoI that a particular mz falls into
# If it falls into nothing, return None
# mz_tol is the window above and below the 
# mean_mz of the RoI. E.g. if mz_tol = 1 Da, then it looks
# plus and minus 1Da
def match(mz, roi_list, mz_tol, mz_units='Da'):
    if len(roi_list) == 0:
        return None
    pos = bisect.bisect_right(roi_list, mz)

    if pos == len(roi_list):
        if mz_units == 'Da':
            dist_left = mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()
        else:  # ppm
            dist_left = 1e6 * (mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()) / mz.get_mean_mz()

        if dist_left < mz_tol:
            return roi_list[pos - 1]
        else:
            return None
    elif pos == 0:
        if mz_units == 'Da':
            dist_right = roi_list[pos].get_mean_mz() - mz.get_mean_mz()
        else:  # ppm
            dist_right = 1e6 * (roi_list[pos].get_mean_mz() - mz.get_mean_mz()) / mz.get_mean_mz()

        if dist_right < mz_tol:
            return roi_list[pos]
        else:
            return None
    else:
        if mz_units == 'Da':
            dist_left = mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()
            dist_right = roi_list[pos].get_mean_mz() - mz.get_mean_mz()
        else:  # ppm
            dist_left = 1e6 * (mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()) / mz.get_mean_mz()
            dist_right = 1e6 * (roi_list[pos].get_mean_mz() - mz.get_mean_mz()) / mz.get_mean_mz()

        if dist_left < mz_tol and dist_right > mz_tol:
            return roi_list[pos - 1]
        elif dist_left > mz_tol and dist_right < mz_tol:
            return roi_list[pos]
        elif dist_left < mz_tol and dist_right < mz_tol:
            if dist_left <= dist_right:
                return roi_list[pos - 1]
            else:
                return roi_list[pos]
        else:
            return None


def roi_correlation(roi1, roi2, min_rt_point_overlap=5, method='pearson'):
    # flip around so that roi1 starts earlier (or equal)
    if roi2.rt_list[0] < roi1.rt_list[0]:
        temp = roi2
        roi2 = roi1
        roi1 = temp

    # check that they meet the min_rt_point overlap
    if roi1.rt_list[-1] < roi2.rt_list[0]:
        # no overlap at all
        return 0.0

    # find the position of the first element in roi2 in roi1
    pos = roi1.rt_list.index(roi2.rt_list[0])

    # print roi1.rt_list
    # print roi2.rt_list
    # print pos

    total_length = max([len(roi1.rt_list), len(roi2.rt_list) + pos])
    # print total_length

    r1 = np.zeros((total_length), np.double)
    r2 = np.zeros_like(r1)

    r1[:len(roi1.rt_list)] = roi1.intensity_list
    r2[pos:pos + len(roi2.rt_list)] = roi2.intensity_list

    # print 
    # for i,a in enumerate(r1):
    #     print "{:10.4f}\t{:10.4f}".format(a,r2[i])
    if method == 'pearson':
        r, _ = pearsonr(r1, r2)
    else:
        r = cosine_score(r1, r2)

    return r


def cosine_score(u, v):
    numerator = (u * v).sum()
    denominator = np.sqrt((u * u).sum()) * np.sqrt((v * v).sum())
    return numerator / denominator


# Make the RoI from an input file
# mz_units = Da for Daltons
# mz_units = ppm for ppm
def make_roi(input_file, mz_tol=0.001, mz_units='Da', min_length=10, min_intensity=50000, start_rt=0, stop_rt=10000000,length_units = "scans"):
    # input_file = 'Beer_multibeers_1_fullscan1.mzML'

    if not mz_units == 'Da' and not mz_units == 'ppm':
        logger.warning("Unknown mz units, use Da or ppm")
        return None, None

    run = pymzml.run.Reader(input_file, MS1_Precision=5e-6,
                            extraAccessions=[('MS:1000016', ['value', 'unitName'])],
                            obo_version='4.0.1')

    live_roi = []
    dead_roi = []
    junk_roi = []

    for spectrum in run:
        # print spectrum['centroid_peaks']
        if spectrum['ms level'] == 1:
            live_roi.sort()
            # current_ms1_scan_rt, units = spectrum['scan start time'] # this no longer works
            current_ms1_scan_rt, units = spectrum.scan_time
            if units == 'minute':
                current_ms1_scan_rt *= 60.0

            if current_ms1_scan_rt < start_rt:
                continue
            if current_ms1_scan_rt > stop_rt:
                break

            # print current_ms1_scan_rt
            # print spectrum.peaks
            not_grew = set(live_roi)
            for mz, intensity in spectrum.peaks('raw'):
                if intensity >= min_intensity:
                    match_roi = match(Roi(mz, 0, 0), live_roi, mz_tol, mz_units=mz_units)
                    if match_roi:
                        match_roi.add(mz, current_ms1_scan_rt, intensity)
                        if match_roi in not_grew:
                            not_grew.remove(match_roi)
                    else:
                        bisect.insort_right(live_roi, Roi(mz, current_ms1_scan_rt, intensity))

            for roi in not_grew:
                if length_units == "scans":
                    if roi.n >= min_length:
                        dead_roi.append(roi)
                    else:
                        junk_roi.append(roi)
                else:
                    if roi.length_in_seconds >= min_length:
                        dead_roi.append(roi)
                    else:
                        junk_roi.append(roi)
                pos = live_roi.index(roi)
                del live_roi[pos]

            # logger.debug("Scan @ {}, {} live ROIs".format(current_ms1_scan_rt, len(live_roi)))

    # process all the live ones - keeping only those that 
    # are longer than the minimum length
    good_roi = dead_roi
    for roi in live_roi:
        if roi.n >= min_length:
            good_roi.append(roi)
        else:
            junk_roi.append(roi)
    return good_roi, junk_roi


def greedy_roi_cluster(roi_list, corr_thresh=0.75, corr_type='cosine'):
    # sort in descending intensity
    roi_list_copy = [r for r in roi_list]
    roi_list_copy.sort(key=lambda x: max(x.intensity_list), reverse=True)
    roi_clusters = []
    while len(roi_list_copy) > 0:
        roi_clusters.append([roi_list_copy[0]])
        remove_idx = [0]
        if len(roi_list_copy) > 1:
            for i, r in enumerate(roi_list_copy[1:]):
                corr = roi_correlation(roi_list_copy[0], r)
                if corr > corr_thresh:
                    roi_clusters[-1].append(r)
                    remove_idx.append(i + 1)
        remove_idx.sort(reverse=True)
        for r in remove_idx:
            del roi_list_copy[r]

    return roi_clusters


class RoiToChemicalCreator(ChemicalCreator):
    """
    Turns ROI to Chemical objects
    """

    def __init__(self, peak_sampler, all_roi, n_peaks=1):
        super().__init__(peak_sampler)
        self.rois_data = all_roi
        self.ms_levels = 2
        self.crp_samples = [[] for i in range(self.ms_levels)]
        self.crp_index = [[] for i in range(self.ms_levels)]
        self.alpha = math.inf
        self.counts = [[] for i in range(self.ms_levels)]
        if self.ms_levels > 2:
            logger.warning(
                "Warning ms_level > 3 not implemented properly yet. Uses scaled ms_level = 2 information for now")

        self.chromatograms = []
        self.chemicals = []
        for i in range(len(self.rois_data)):
            if i % 50000 == 0:
                logger.debug('%6d/%6d' % (i, len(self.rois_data)))
            roi = self.rois_data[i]

            # raise numpy warning as exception, see https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
            chrom = None
            with np.errstate(divide='raise'):
                try:
                    chrom = roi.to_chromatogram()
                except FloatingPointError:
                    logger.debug('Invalid chromatogram {}'.format(i))
                except ZeroDivisionError:
                    logger.debug('Invalid chromatogram {}'.format(i))

            if chrom is not None:
                chem = self._to_unknown_chemical(chrom)
                if self.peak_sampler is not None:
                    try:
                        # TODO: initialise chemical with only 1 child for the purpose of experiment, we might need to improve this
                        chem.children = self._get_children(GET_MS2_BY_PEAKS, chem, n_peaks=n_peaks)
                    except KeyError:
                        pass
                self.chromatograms.append(chrom)
                self.chemicals.append(chem)
        assert len(self.chromatograms) == len(self.chemicals)
        logger.info('Found %d ROIs above thresholds' % len(self.chromatograms))

    def sample(self, chromatogram_creator, mz_range, rt_range, min_ms1_intensity, n_ms1_peaks, ms_levels=2,
               chemical_type=None,
               formula_list=None, compound_list=None, alpha=math.inf, fixed_mz=False, adduct_proportion_cutoff=0.05):
        return NotImplementedError()

    def sample_from_chromatograms(self, chromatogram_creator, min_rt, max_rt, min_ms1_intensity, ms_levels=2):
        return NotImplementedError()

    def _to_unknown_chemical(self, chrom):
        idx = np.argmax(chrom.raw_intensities)  # find intensity apex
        mz = chrom.raw_mzs[idx]

        # In the MassSpec, we assume that chemical starts eluting from chem.rt + chem.chromatogram.rts (normalised to start from 0)
        # So here, we have to set set chemical rt to start from the minimum of chromatogram raw rts, so it elutes correct.
        # rt = chrom.raw_rts[idx]
        rt = min(chrom.raw_rts)

        max_intensity = chrom.raw_intensities[idx]
        mz = mz - PROTON_MASS
        chem = UnknownChemical(mz, rt, max_intensity, chrom, None)
        chem.type = CHEM_NOISE
        return chem

    def plot_chems(self, n_plots, reverse=False):
        sorted_chems = sorted(self.chemicals, key=lambda chem: chem.chromatogram.roi.num_scans())
        if reverse:
            sorted_chems.reverse()
        for c in sorted_chems[0:n_plots]:
            chrom = c.chromatogram
            plt.plot(chrom.raw_rts, chrom.raw_intensities)
            plt.show()


def plot_roi(roi, statuses=None, log=False):
    if log:
        intensities = np.log(roi.intensity_list)
        plt.ylabel('Log Intensity')
    else:
        intensities = roi.intensity_list
        plt.ylabel('Intensity')
    if statuses is not None:
        colours = []
        for s in statuses:
            if s == 'Noise':
                colours.append('red')
            elif s == 'Increase':
                colours.append('blue')
            elif s == 'Decrease':
                colours.append('yellow')
            else:
                colours.append('green')
        plt.scatter(roi.rt_list, intensities, color=colours)
    else:
        plt.scatter(roi.rt_list, intensities)
    plt.xlabel('RT')
    plt.show()



