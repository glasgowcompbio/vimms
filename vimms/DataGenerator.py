import copy
import glob
import os
import xml.etree.ElementTree
import zipfile

import numpy as np
import pandas as pd
import pylab as plt
import pymzml
from loguru import logger
from sklearn.neighbors import KernelDensity

from vimms.Chemicals import DatabaseCompound
from vimms.Common import MZ, INTENSITY, RT, N_PEAKS, SCAN_DURATION, MZ_INTENSITY_RT, save_obj
from vimms.MassSpec import Peak, Scan
from vimms.SpectralUtils import get_precursor_info


def extract_hmdb_metabolite(in_file, delete=True):
    logger.debug('Extracting HMDB metabolites from %s' % in_file)

    # if out_file is zipped then extract the xml file inside
    try:
        # extract from zip file
        zf = zipfile.ZipFile(in_file, 'r')
        metabolite_xml_file = zf.namelist()[0]  # assume there's only a single file inside the zip file
        f = zf.open(metabolite_xml_file)
    except zipfile.BadZipFile:  # oops not a zip file
        zf = None
        f = in_file

    # loops through file and extract the necessary element text to create a DatabaseCompound
    db = xml.etree.ElementTree.parse(f).getroot()
    compounds = []
    prefix = '{http://www.hmdb.ca}'
    for metabolite_element in db:
        row = [None, None, None, None, None, None]
        for element in metabolite_element:
            if element.tag == (prefix + 'name'):
                row[0] = element.text
            elif element.tag == (prefix + 'chemical_formula'):
                row[1] = element.text
            elif element.tag == (prefix + 'monisotopic_molecular_weight'):
                row[2] = element.text
            elif element.tag == (prefix + 'smiles'):
                row[3] = element.text
            elif element.tag == (prefix + 'inchi'):
                row[4] = element.text
            elif element.tag == (prefix + 'inchikey'):
                row[5] = element.text

        # if all fields are present, then add them as a DatabaseCompound
        if None not in row:
            compound = DatabaseCompound(row[0], row[1], row[2], row[3], row[4], row[5])
            compounds.append(compound)
    logger.info('Loaded %d DatabaseCompounds from %s' % (len(compounds), in_file))

    f.close()
    if zf is not None:
        zf.close()

    if delete:
        logger.info('Deleting %s' % in_file)
        os.remove(in_file)

    return compounds


def get_data_source(mzml_path, filename, xcms_output=None):
    """
    Load a `DataSource` object that stores information on a set of .mzML files.
    :param mzml_path: the location of .mzML files to train the KDEs.
    :param filename: a particular .mzML file to be used. If None then all files in `mzml_path` will be used.
    :param xcms_output: As an option, we can use XCMS peak picking results to train the (mz, RT, intensity) densities.
    This makes the generated spectra more similar to real ones after peak picking. If not available, leave this as None.
    :return: a DataSource object.
    """
    ds = DataSource()
    ds.load_data(mzml_path, filename)
    if xcms_output is not None:
        ds.load_xcms_output(xcms_output)
    return ds


def get_spectral_feature_database(ds, filename, min_ms1_intensity, min_ms2_intensity, min_rt, max_rt,
                                  bandwidth_mz_intensity_rt, bandwidth_n_peaks, out_file=None):
    """
    Generate spectral feature database on the .mzML files that have been loaded into the DataSource
    :param ds: the `DataSource` object that contains loaded .mzML files.
    :param filename: a particular .mzML file to be used. If None then all loaded files in `ds` will be used.
    :param min_ms1_intensity: minimum MS1 intensity to include a data point to train the KDEs.
    :param min_ms2_intensity: minimum MS2 intensity to include a data point to train the KDEs.
    :param min_rt: minimum RT to include a data point to train the KDEs.
    :param max_rt: maximum RT to include a data point to train the KDEs.
    :param bandwidth_mz_intensity_rt: the bandwidth of the kernel to train the KDEs for (mz, RT, intensity) values.
    :param bandwidth_n_peaks: the bandwidth of the kernel to train the KDEs for the number of peaks per scan.
    :param out_file: the resulting output file to store the trained KDEs (in form of `PeakSampler` object).
    :return: a PeakSampler object that can be used to draw samples for simulation.
    """
    ps = PeakSampler(ds, min_rt, max_rt, min_ms1_intensity, min_ms2_intensity, filename, False,
                     bandwidth_mz_intensity_rt, bandwidth_n_peaks)
    if out_file is not None:
        save_obj(ps, out_file)
    return ps


def filter_df(df, min_ms1_intensity, rt_range, mz_range):
    # filter by rt range
    if rt_range is not None:
        df = df[(df['rt'] > rt_range[0][0]) & (df['rt'] < rt_range[0][1])]

    # filter by mz range
    if mz_range is not None:
        df = df[(df['rt'] > mz_range[0][0]) & (df['rt'] < mz_range[0][1])]

    # filter by min intensity
    intensity_col = 'maxo'
    if min_ms1_intensity is not None:
        df = df[(df[intensity_col] > min_ms1_intensity)]
    return df


class DataSource(object):
    """
    A class to load and extract centroided peaks from CSV and mzML files.
    :param min_ms1_intensity: minimum ms1 intensity for filtering
    :param min_ms2_intensity: maximum ms2 intensity for filtering
    :param min_rt: minimum RT for filtering
    :param max_rt: maximum RT for filtering
    """

    def __init__(self):
        # A dictionary that stores the actual pymzml spectra for each filename
        self.file_spectra = {}  # key: filename, value: a dict where key is scan_number and value is spectrum

        # A dictionary to store the distribution on scan durations for each ms_level in each file
        self.file_scan_durations = {}  # key: filename, value: a dict with key ms level and value scan durations

        # A dictionary to store extracted MS2 scans
        self.precursor_info = {}  # key: filename, value: a dataframe of precursor info

        # pymzml parameters
        self.ms1_precision = 5e-6
        self.obo_version = '4.0.1'

        # xcms peak picking results, if any
        self.df = None

    def load_data(self, mzml_path, file_name=None):
        """
        Loads data and generate peaks from mzML files. The resulting peak objects will not have chromatographic peak
        shapes, because no peak picking has been performed yet.
        :param mzml_path: the input folder containing the mzML files
        :return: nothing, but the instance variable file_spectra and scan_durations are populated
        """
        for filename in glob.glob(os.path.join(mzml_path, '*.mzML')):
            fname = os.path.basename(filename)
            if file_name is not None and fname != file_name:
                continue
            logger.info('Loading %s' % fname)

            # TODO: inefficient because we have to parse the mzML file multiple times
            self.file_spectra[fname] = self.extract_all_scans(filename)
            self.precursor_info[fname] = self.extract_precursor_info(filename)
            self.file_scan_durations[fname] = self.extract_scan_durations(filename)

    def extract_all_scans(self, filename):
        scans = {}
        run = pymzml.run.Reader(filename, obo_version=self.obo_version,
                                MS1_Precision=self.ms1_precision,
                                extraAccessions=[('MS:1000016', ['value', 'unitName'])])
        for scan_no, scan in enumerate(run):
            scans[scan_no] = scan
        return scans

    def extract_precursor_info(self, filename):
        df = get_precursor_info(filename)
        return df

    def extract_scan_durations(self, filename):
        transitions = {
            (1, 1): [],
            (1, 2): [],
            (2, 1): [],
            (2, 2): []
        }
        run = pymzml.run.Reader(filename, obo_version=self.obo_version,
                                MS1_Precision=self.ms1_precision,
                                extraAccessions=[('MS:1000016', ['value', 'unitName'])])
        for scan_no, scan in enumerate(run):
            if scan_no == 0:
                previous_level = scan['ms level']
                old_rt = self._get_rt(scan)
                continue
            rt = self._get_rt(scan)
            current_level = scan['ms level']
            previous_duration = rt - old_rt
            transitions[(previous_level, current_level)].append(previous_duration)
            previous_level = current_level
            old_rt = rt
        return transitions

    def load_xcms_output(self, xcms_filename):
        self.df = pd.read_csv(xcms_filename)

    def plot_data(self, file_name, ms_level=1, min_rt=None, max_rt=None, max_data=100000):
        data_types = [MZ, INTENSITY, RT, N_PEAKS, SCAN_DURATION]
        for data_type in data_types:
            if data_type == SCAN_DURATION:
                X = self.get_scan_durations(file_name)
                self.plot_histogram(X, data_type)
            elif data_type == N_PEAKS:
                X = self.get_n_peaks(file_name, ms_level, min_rt=min_rt, max_rt=max_rt)
            else:
                X = self.get_data(data_type, file_name, ms_level, min_rt=min_rt, max_rt=max_rt, max_data=max_data)
                if data_type == INTENSITY:
                    X = np.log(X)
                self.plot_histogram(X, data_type)
                self.plot_boxplot(X, data_type)

    def plot_histogram(self, X, data_type, n_bins=100):
        """
        Makes a histogram plot on the distribution of the item of interest
        :param X: a numpy array
        :param bins: number of histogram bins
        :return: nothing. A plot is shown.
        """
        if data_type == SCAN_DURATION:
            rt_steps = X
            for key, rt_list in rt_steps.items():
                try:
                    bins = np.linspace(min(rt_list), max(rt_list), n_bins)
                    plt.figure()
                    plt.hist(rt_list, bins=bins)
                    plt.title(key)
                    plt.show()
                except ValueError:
                    continue
        else:
            plt.figure()
            _ = plt.hist(X, bins=n_bins)
            plt.plot(X[:, 0], np.full(X.shape[0], -0.01), '|k')
            plt.title('Histogram for %s -- shape %s' % (data_type, str(X.shape)))
            plt.show()

    def plot_boxplot(self, X, data_type):
        """
        Makes a boxplot on the distribution of the item of interest
        :param X: a numpy array
        :return: nothing. A plot is shown.
        """
        plt.figure()
        _ = plt.boxplot(X)
        plt.title('Boxplot for %s -- shape %s' % (data_type, str(X.shape)))
        plt.show()

    def plot_peak(self, peak):
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(peak.rt_values, peak.intensity_values)
        axarr[1].plot(peak.rt_values, peak.mz_values, linestyle='None', marker='o', markersize=1.0, color='b')

    def get_data(self, data_type, filename, ms_level, min_intensity=None,
                 min_rt=None, max_rt=None, log=False, max_data=100000):
        """
        Retrieves values as numpy array
        :param data_type: data_type is 'mz', 'rt', 'intensity' or 'n_peaks'
        :param filename: the mzml filename or None for all files
        :param ms_level: level 1 or 2
        :param min_intensity: minimum ms2 intensity for thresholding
        :param min_rt: minimum RT value for thresholding
        :param max_rt: max RT value for thresholding
        :param log: if true, the returned values will be logged
        :return: an Nx1 numpy array of all the values requested
        """
        # if xcms peak picking results are provided, use that instead
        if ms_level == 1 and self.df is not None:
            logger.info('Using values from XCMS peaklist')

            # remove rows in the peak picked dataframe that are outside threshold values
            df = filter_df(self.df, min_intensity, [[min_rt, max_rt]], None)

            # extract the values we need
            if data_type == MZ:
                X = df['mz'].values
            elif data_type == RT:
                # we use rt for the starting value for the chemical to elute
                X = df['rt'].values
            elif data_type == INTENSITY:
                X = df['maxo'].values
            elif data_type == MZ_INTENSITY_RT:
                X = df[['mz', 'maxo', 'rt']].values

        else:  # else we get the values by reading from the scans in mzML files directly
            logger.info('Using values from scans')

            # get spectra from either one file or all files
            if filename is None:  # use all spectra
                all_spectra = []
                for f in self.file_spectra:
                    spectra_for_f = list(self.file_spectra[f].values())
                    all_spectra.extend(spectra_for_f)
            else:  # use spectra for that file only
                all_spectra = self.file_spectra[filename].values()

            # loop through spectrum and get all peaks above threshold
            values = []
            for spectrum in all_spectra:
                # if wrong ms level, skip this spectrum
                if spectrum.ms_level != ms_level:
                    continue

                # collect all valid Peak objects in a spectrum
                spectrum_peaks = []
                for mz, intensity in spectrum.peaks('raw'):
                    rt = self._get_rt(spectrum)
                    p = Peak(mz, rt, intensity, spectrum.ms_level)
                    if self._valid_peak(p, min_intensity, min_rt, max_rt):
                        spectrum_peaks.append(p)

                if data_type == MZ_INTENSITY_RT:  # used when fitting m/z, rt and intensity together for the manuscript
                    mzs = list(getattr(x, MZ) for x in spectrum_peaks)
                    intensities = list(getattr(x, INTENSITY) for x in spectrum_peaks)
                    rts = list(getattr(x, RT) for x in spectrum_peaks)
                    values.extend(list(zip(mzs, intensities, rts)))

                else:  # MZ, INTENSITY or RT separately
                    attrs = list(getattr(x, data_type) for x in spectrum_peaks)
                    values.extend(attrs)

            X = np.array(values)

        # log-transform if necessary
        if log:
            if data_type == MZ_INTENSITY_RT:  # just log the intensity part
                X[:, 1] = np.log(X[:, 1])
            else:
                X = np.log(X)

        # pick random samples
        try:
            idx = np.arange(len(X))
            rnd_idx = np.random.choice(idx, size=int(max_data), replace=False)
            sampled_X = X[rnd_idx]
        except ValueError:
            sampled_X = X

        # return values
        if data_type == MZ_INTENSITY_RT:
            return sampled_X  # it's already a Nx2 or Nx3 array
        else:
            # convert into Nx1 array
            return sampled_X[:, np.newaxis]

    def get_n_peaks(self, filename, ms_level, min_intensity=None, min_rt=None, max_rt=None):
        # get spectra from either one file or all files
        if filename is None:  # use all spectra
            all_spectra = []
            for f in self.file_spectra:
                spectra_for_f = list(self.file_spectra[f].values())
                all_spectra.extend(spectra_for_f)
        else:  # use spectra for that file only
            all_spectra = self.file_spectra[filename].values()

        # loop through spectrum and get all peaks above threshold
        values = []
        for spectrum in all_spectra:
            # if wrong ms level, skip this spectrum
            if spectrum.ms_level != ms_level:
                continue

            # collect all valid Peak objects in a spectrum
            spectrum_peaks = []
            for mz, intensity in spectrum.peaks('raw'):
                rt = self._get_rt(spectrum)
                p = Peak(mz, rt, intensity, spectrum.ms_level)
                if self._valid_peak(p, min_intensity, min_rt, max_rt):
                    spectrum_peaks.append(p)

            # collect the data points we need into a list
            n_peaks = len(spectrum_peaks)
            if n_peaks > 0:
                values.append(n_peaks)

        # convert into Nx1 array
        X = np.array(values)
        return X[:, np.newaxis]

    def get_scan_durations(self, fname):
        if fname is None:  # if no filename, then combine all the dictionary keys
            combined = None
            for f in self.file_scan_durations:
                if combined is None:  # copy first one
                    combined = copy.deepcopy(self.file_scan_durations[f])
                else:  # and extend with the subsequent ones
                    for key in combined:
                        combined[key].extend(self.file_scan_durations[f][key])
        else:
            combined = self.file_scan_durations[fname]
        return combined

    def _get_rt(self, spectrum):
        rt, units = spectrum.scan_time
        if units == 'minute':
            rt *= 60.0
        return rt

    def _valid_peak(self, peak, min_intensity, min_rt, max_rt):
        if min_intensity is not None and peak.intensity < min_intensity:
            return False
        elif min_rt is not None and peak.rt < min_rt:
            return False
        elif max_rt is not None and peak.rt > max_rt:
            return False
        else:
            return True


class PeakSampler(object):
    """A class to sample peaks from a trained density estimator"""

    # TODO: add min intensity threshold here so we don't store everything??!!!
    def __init__(self, data_source, min_rt, max_rt, min_ms1_intensity, min_ms2_intensity,
                 filename=None, plot=False,
                 bandwidth_mz_intensity_rt=1.0, bandwidth_n_peaks=1.0, filename_to_N_DEW=None):
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_ms1_intensity = min_ms1_intensity
        self.min_ms2_intensity = min_ms2_intensity
        self.filename = filename
        self.plot = plot
        self.filename_to_N_DEW = filename_to_N_DEW  # a dictionary that maps from filename to (N, DEW)

        # get all the scan dataframes across all files and combine them all
        self.all_ms2_scans = self._extract_ms2_scans(data_source)
        logger.debug('Extracted %d MS2 scans' % len(self.all_ms2_scans))

        # compute sum(ms2 peak intensities) / ms1.intensity
        self.intensity_props = self._compute_intensity_props()

        # extract scan durations
        self.file_scan_durations = {}  # key: (N, DEW), value: a list of scan durations for (N, DEW)
        if filename_to_N_DEW is None:
            # no mapping between filename to N is specified, so we just assign it a default key of 0
            logger.debug('Extracting scan durations')
            N_DEW = (0, 0)  # default value if not specified
            self.file_scan_durations[N_DEW] = data_source.get_scan_durations(filename)
        else:
            # store the scan durations for the different Ns
            for filename, v in filename_to_N_DEW.items():
                N, DEW = v
                logger.debug('Extracting scan durations for N=%d DEW=%d from %s' % (N, DEW, filename))
                self.file_scan_durations[v] = data_source.get_scan_durations(filename)

        # train KDEs for each ms-level
        max_data = 100000
        self.kdes = {}
        self.kernel = 'gaussian'
        self._kde(data_source, filename, 1, bandwidth_mz_intensity_rt, bandwidth_n_peaks, max_data)
        try:  # exceptions if data_source only contains fullscan data but we try to train kde on ms level 2
            self._kde(data_source, filename, 2, bandwidth_mz_intensity_rt, bandwidth_n_peaks, max_data)
        except ValueError:
            pass
        except IndexError:
            pass

    ####################################################################################################################
    # Public methods
    ####################################################################################################################

    def scan_durations(self, previous_level, current_level, n_sample, N, DEW):
        # the scan durations is stored for each N and DEW combination
        key = (previous_level, current_level,)
        try:
            file_scan_durations = self.file_scan_durations[(N, DEW)]
            values = file_scan_durations[key]
        except KeyError:  # if (N, DEW) not found in self.file_scan_durations

            # if we only have one pair of (N, DEW) then use that as a default
            if len(self.file_scan_durations) == 1:
                selected = list(self.file_scan_durations.keys())[0]
                file_scan_durations = self.file_scan_durations[selected]
                values = file_scan_durations[key]

            # if there are multiple (N, DEW) values, then pick the closest
            else:
                nodes = list(self.file_scan_durations.keys())
                try:
                    nodes.remove((0, 0))
                except ValueError:
                    pass
                nodes = np.asarray(nodes)
                node = np.array((N, DEW))
                dist = np.sum((nodes - node) ** 2, axis=1)
                pos = np.argmin(dist)
                selected = tuple(nodes[pos])
                file_scan_durations = self.file_scan_durations[selected]
                values = file_scan_durations[key]

            msg = 'No scan durations for (N=%d, DEW=%d), using (N=%d, DEW=%d) instead' % (
                N, DEW, selected[0], selected[1])
            logger.debug(msg)

        if len(values) == 0:  # if values are empty, then we just return an empty array
            return np.array([])
        elif len(values) < n_sample:  # if not enough values, then return them all
            return values
        else:  # sample scan durations without replacement
            try:
                sampled = np.random.choice(values, replace=False, size=n_sample)
                return sampled
            except ValueError:
                return np.array([])

    def get_peak(self, ms_level, N=None, min_mz=None, max_mz=None, min_rt=None, max_rt=None, min_intensity=None):
        if N is None:
            N = max(self.n_peaks(ms_level, 1).astype(int)[0][0], 0)

        peaks = []
        while len(peaks) < N:
            vals = self.sample(ms_level, 1)
            intensity = np.exp(vals[0, 1])
            mz = vals[0, 0]
            rt = vals[0, 2]
            p = Peak(mz, rt, intensity, ms_level)
            if self._is_valid(p, min_mz, max_mz, min_rt, max_rt, min_intensity):  # othwerise we just keep rejecting
                peaks.append(p)
        return peaks

    def sample(self, ms_level, n_sample):
        vals = self.kdes[(MZ_INTENSITY_RT, ms_level)].sample(n_sample)
        return vals

    def n_peaks(self, ms_level, n_sample):
        return self.kdes[(N_PEAKS, ms_level)].sample(n_sample)

    def get_ms2_spectra(self, N=1):
        spectra = []
        total = len(self.all_ms2_scans)

        if total > 0:
            # select N random spectra
            idx = np.random.choice(total, replace=False, size=N)
            samples = self.all_ms2_scans.iloc[idx, :]

            # convert to Scan objects
            for idx, row in samples.iterrows():
                # create precursor (MS1) peak
                parent_ms_level = 1
                parent_mz = row['ms1_mz']
                parent_rt = row['ms1_scan_rt']
                parent_intensity = row['ms1_intensity']
                parent_peak = Peak(parent_mz, parent_rt, parent_intensity, parent_ms_level)

                # create MS2 scan
                ms_level = 2
                ms2_peaks = row['ms2_peaklist']
                ms2_scan_id = idx
                ms2_mzs = ms2_peaks[:, 0]
                ms2_rt = ms2_peaks[0, 1]  # all the values are the same, so we can take the first one
                ms2_intensities = ms2_peaks[:, 2]  # TODO: filter by min_ms2_intensity here
                ms2_scan = Scan(ms2_scan_id, ms2_mzs, ms2_intensities, ms_level, ms2_rt, parent=parent_peak)
                spectra.append(ms2_scan)
        return spectra

    def get_noise_sample(self):
        # TODO: finish this
        # need to choose number of noise fragments from get_num_noisy_samples() below
        # then draw n noise fragments
        # returns list of ms2 noise fragments. type = MSN
        # noise fragment here is defined as ms2 peaks below some intensity threshold
        return []

    def get_num_noisy_samples(self):
        # TODO: finish this
        # returns a distribution of the number of noise fragments
        return 0.0

    def get_msn_noisy_intensity(self, intensity, ms_level):
        # TODO: until we characterise the noise properly, just return the original value for now
        # takes intensity
        # adds noise, but ensures its positive value
        # returns list with one numeric value
        # ignores ms_level for now
        return intensity

    def get_msn_noisy_mz(self, mz, ms_level):
        # TODO: finish this
        # same as above, but for m/z
        # Simon: We can characterise mz noise from the chromatographic peaks we extract.
        # I suggest a constant variance for now, but we might want to fit models where we account for
        # variability in noise variance as a function of mz itself, and intensity.
        return mz

    def get_parent_intensity_proportion(self, N=1):
        # this is the proportion of all fragment intensities in a spectra over the parent intensity
        # returns number between 0 and 1
        if len(self.all_ms2_scans) > 0:
            prop = np.random.choice(self.intensity_props, replace=False, size=N)
            if N == 1:  # flatten so that it isn't an array
                prop = prop[0]
            return prop
        return None

    ####################################################################################################################
    # Private methods used in the constructor
    ####################################################################################################################

    def _extract_ms2_scans(self, data_source):
        combined_dfs = pd.concat(data_source.precursor_info.values())
        # select only the column we need
        # 'ms2_peaklist' is a 2d-array, where each row is an ms2 peak, and columns are mz, rt, intensity
        col_names = ['ms1_mz', 'ms1_scan_rt', 'ms1_intensity', 'ms2_peaklist']
        return combined_dfs[col_names]

    def _compute_intensity_props(self):
        logger.debug('Computing parent intensity proportions')
        intensity_props = []
        for idx, row in self.all_ms2_scans.iterrows():
            parent_intensity = row['ms1_intensity']
            ms2_peaks = row['ms2_peaklist']
            ms2_intensities = ms2_peaks[:, 2]
            prop = np.sum(ms2_intensities) / parent_intensity
            if prop <= 1:
                intensity_props.append(prop)
        return np.array(intensity_props)

    def _kde(self, data_source, filename, ms_level, bandwidth_mz_intensity_rt, bandwidth_n_peaks, max_data):
        logger.debug('Training KDEs for ms_level=%d' % ms_level)
        params = [
            {'data_type': MZ_INTENSITY_RT, 'bandwidth': bandwidth_mz_intensity_rt},
            {'data_type': N_PEAKS, 'bandwidth': bandwidth_n_peaks}
        ]

        for param in params:
            data_type = param['data_type']
            min_intensity = self.min_ms1_intensity if ms_level == 1 else self.min_ms2_intensity

            # get data
            logger.debug('Retrieving %s values from %s' % (data_type, data_source))
            if data_type == N_PEAKS:
                X = data_source.get_n_peaks(filename, ms_level, min_intensity=min_intensity,
                                            min_rt=self.min_rt, max_rt=self.max_rt)
            else:
                log = True if data_type == MZ_INTENSITY_RT else False
                X = data_source.get_data(data_type, filename, ms_level, min_intensity=min_intensity,
                                         min_rt=self.min_rt, max_rt=self.max_rt, log=log, max_data=max_data)

            # fit kde
            bandwidth = param['bandwidth']
            kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(X)
            self.kdes[(data_type, ms_level)] = kde

            # plot if necessary
            self._plot(kde, X, data_type, filename, bandwidth)

    def _is_valid(self, peak, min_mz, max_mz, min_rt, max_rt, min_intensity):
        if peak.intensity < 0:
            return False
        if min_mz is not None and min_mz > peak.mz:
            return False
        if max_mz is not None and max_mz < peak.mz:
            return False
        if min_rt is not None and min_rt > peak.rt:
            return False
        if max_rt is not None and max_rt < peak.rt:
            return False
        if min_intensity is not None and min_intensity > peak.intensity:
            return False
        return True

    def _plot(self, kde, X, data_type, filename, bandwidth):
        if self.plot:
            if data_type == MZ_INTENSITY_RT:
                logger.debug('3D plotting for %s not implemented' % MZ_INTENSITY_RT)
            else:
                fname = 'All' if filename is None else filename
                title = '%s density estimation for %s - bandwidth %.3f' % (data_type, fname, bandwidth)
                X_plot = np.linspace(np.min(X), np.max(X), 1000)[:, np.newaxis]
                log_dens = kde.score_samples(X_plot)
                plt.figure()
                plt.fill_between(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
                plt.plot(X[:, 0], np.full(X.shape[0], -0.01), '|k')
                plt.title(title)
                plt.show()
