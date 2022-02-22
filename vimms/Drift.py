import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


###################################################################################################
# Simple Example Data Classes
###################################################################################################


class SimpleScan():
    def __init__(self, start_time, end_time, sample_id, ms_level):
        self.start_time = start_time
        self.end_time = end_time
        self.sample_id = sample_id
        self.ms_level = ms_level
        self.mzs = []
        self.rts = []
        self.peak_ids = []
        self.peak_statuses = []

    def add(self, peak):
        if self.ms_level == 1:
            self.mzs.append(peak.mz)
            self.rts.append(peak.rt)
            self.peak_ids.append(peak.id)
            self.peak_statuses.append(peak.peak_status)
        else:
            self.mzs.append(peak.spectra)
            self.rts.append(peak.rt)
            self.peak_ids.append(peak.id)
            self.peak_statuses.append(peak.peak_status)


class SimplePeak():
    def __init__(self, mz, rt, sample_id, peak_id, peak_status, est_drift,
                 est_drift_sd):
        self.mzs = [mz]
        self.rts = [rt]
        self.sample_ids = [sample_id]
        self.peak_ids = [peak_id]
        self.peak_statuses = [peak_status]
        self.est_drifts = [est_drift]
        self.est_drifts_sd = [est_drift_sd]
        self.est_rt = rt
        self.est_mz = mz
        self.frag_scans = []
        self.spectra = None

    def add_ms1_scan(self, mz, rt, sample_id, peak_id, peak_status, est_drift,
                     est_drift_sd):
        self.mzs.append(mz)
        self.rts.append(rt)
        self.sample_ids.append(sample_id)
        self.peak_ids.append(peak_id)
        self.peak_statuses.append(peak_status)
        self.est_drifts.append(est_drift)
        self.est_drifts_sd.append(est_drift_sd)
        self.update()

    def add_ms2_scan(self, ms2_scan):
        self.frag_scans.append(ms2_scan)
        # self.update_spectra(ms2_scan)

    def remove_last_entry(self):
        self.mzs.pop()
        self.rts.pop()
        self.sample_ids.pop()
        self.peak_ids.pop()
        self.peak_statuses.pop()
        self.est_drifts.pop()
        self.est_drifts_sd.pop()
        self.update()

    # def update_spectra(self, ms2_scan):
    #     if self.spectra is None:
    #         self.spectra = ms2_scan.frag_spectra
    #     elif self.spectra == ms2_scan.frag_spectra:
    #         pass
    #     else:
    #         print('Error: Multiple frag spectra assigned to one anchor')

    def update(self):
        self.est_rt = sum(self.rts) / len(self.rts)
        self.est_mz = sum(self.mzs) / len(self.mzs)


class SimpleChemical():
    def __init__(self, rt, mz, prevalence, id, peak_status):
        self.rt = rt
        self.mz = mz
        self.prevalence = prevalence
        self.id = id  # true id for simulated case
        self.peak_status = peak_status
        self.spectra = id


def get_chems(n_chems, rt_range, mz_range, peak_status=True, prev_range=None,
              start_idx=0):
    chems = []
    for p in range(n_chems):
        rt = np.random.uniform(rt_range[0], rt_range[1], 1)[0]
        mz = np.random.uniform(mz_range[0], mz_range[1], 1)[0]
        if peak_status:
            prev = np.random.uniform(prev_range[0], prev_range[1])
        else:
            prev = None
        idx = start_idx + p
        chem = SimpleChemical(rt, mz, prev, idx, peak_status)
        chems.append(chem)
    return chems


def get_datasets(n_samples, rt_range, n_gp_points, gp_params, data_params,
                 n_chems, mz_range, prev_range, n_noise=0):
    dataset_dict = dict()
    drift_model = DriftSimulator(n_samples, rt_range, n_gp_points, gp_params,
                                 data_params)
    chems = get_chems(n_chems, rt_range, mz_range, prev_range=prev_range)
    noise_idx = n_chems
    for i in range(n_samples):
        dataset = []
        # get chems for current dataset
        for c in chems:
            urv = np.random.uniform(0, 1, 1)
            if c.prevalence > urv:
                drift = drift_model.get_drift(i, c.rt)
                c.rt = c.rt + drift
                new_chem = SimpleChemical(c.rt + drift, c.mz, c.prevalence,
                                          c.id, c.peak_status)
                dataset.append(new_chem)
        # add noise for current dataset
        new_noise = get_chems(n_noise, rt_range, mz_range, False,
                              start_idx=noise_idx)
        noise_idx += n_noise
        dataset.extend(new_noise)
        dataset_dict['dataset' + str(i)] = dataset
    return dataset_dict, drift_model


def plot_dataset(dataset, peak_colour='r', noise_colour='k'):
    rts = [[], []]
    mzs = [[], []]
    for d in dataset:
        rts.append(d.rt)
        mzs.append(d.mz)
        if d.peak_status:
            rts[0].append(d.rt)
            mzs[0].append(d.mz)
        else:
            rts[1].append(d.rt)
            mzs[1].append(d.mz)
    peaks = plt.scatter(rts[0], mzs[0], color=peak_colour)
    noise = plt.scatter(rts[1], mzs[1], color=noise_colour)

    plt.legend((peaks, noise), ('Peaks', 'Noise'), scatterpoints=1,
               loc='lower left')
    plt.ylabel('m/z')
    plt.xlabel('rt')
    plt.show()


def plot_datasets(datasets, colours=['r', 'b', 'g', 'c', 'm', 'y']):
    rts = [[] for d in datasets]
    mzs = [[] for d in datasets]
    keys = list(datasets.keys())
    legend_elements = []
    fig, ax = plt.subplots()
    for i in range(len(datasets)):
        for d in datasets[keys[i]]:
            if d.peak_status:
                rts[i].append(d.rt)
                mzs[i].append(d.mz)
        plt.scatter(rts[i], mzs[i], color=colours[i])
        legend_elements.append(
            Line2D([0], [0], marker='o', color=colours[i], label=keys[i],
                   markerfacecolor='g'))
    ax.legend(handles=legend_elements, loc='lower left')
    plt.ylabel('m/z')
    plt.xlabel('rt')
    plt.show()


###################################################################################################
# Generic classes
###################################################################################################


class DriftSimulator():
    def __init__(self, n_samples, rt_range, n_gp_points, gp_params,
                 data_params):
        self.n_samples = n_samples
        X = np.linspace(rt_range[0], rt_range[1], n_gp_points)[:, np.newaxis]
        y = np.random.normal(data_params[0], data_params[1], n_gp_points)
        kernel = gp_params[0] * RBF(length_scale=gp_params[1],
                                    length_scale_bounds=(1e-1, 10.0))
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.gp.fit(X, y)

    def get_drift(self, model_idx, rt):
        samples = self.gp.sample_y(np.array([[rt]]), n_samples=self.n_samples)
        return samples[0][model_idx]

    def plot_drifts(self):
        NotImplementedError()


class PeakMatching():
    def __init__(self, covariance, max_match_score, ms2_match):
        self.covariance = covariance
        self.max_match_score = max_match_score
        self.ms2_match = ms2_match
        self.anchors = []
        self.possible_anchors = []
        self.matched_anchors = []
        self.X = []
        self.y = []

    def reset_current_statuses(self):
        self.possible_anchors = self.matched_anchors + self.possible_anchors
        [anchor.update() for anchor in self.possible_anchors]
        self.anchors = self.possible_anchors
        self.matched_anchors = []

    def update_xy(self):
        NotImplementedError()

    def add_new_scans(self, scan, est_drift, est_drift_sd):
        NotImplementedError()

    def match_scan(self, scan, est_drift, est_drift_sd):
        NotImplementedError()


###################################################################################################
# Simple Example Drift Method Classes
###################################################################################################


class SimplePeakMatching(PeakMatching):
    def __init__(self, covariance, max_match_score, ms2_match=False):
        super().__init__(covariance, max_match_score, ms2_match)

    def update_xy(self):
        current_rts = np.array(
            [anchor.rts[-1] for anchor in self.matched_anchors])
        anchor_est_rts = np.array(
            [anchor.est_rt for anchor in self.matched_anchors])
        self.X = current_rts[:, np.newaxis]
        self.y = anchor_est_rts - current_rts

    def add_new_scans(self, scans, est_drift, est_drift_sd):
        self.match_scans(scans, est_drift, est_drift_sd)
        self.update_xy()
        saved_dataset = {'X': self.X, 'y': self.y, 't': scans[0].end_time}
        return saved_dataset

    def match_scans(self, scans, est_drift, est_drift_sd):
        for scan in scans:
            if self.ms2_match:
                # create anchors only based on peaks with ms2s
                pass
            else:
                # create anchors with all information, currently ignores ms2 information
                if scan.ms_level == 1:
                    for i in range(len(scan.mzs)):
                        if len(self.possible_anchors) > 0:
                            rt_diff = np.array(
                                [anchor.est_rt - scan.rts[i] for anchor in
                                 self.possible_anchors])
                            mz_diff = np.array(
                                [anchor.est_mz - scan.mzs[i] for anchor in
                                 self.possible_anchors])
                            diff_score = np.array([np.matmul(
                                np.matmul(np.array([rt_diff[i], mz_diff[i]]),
                                          np.linalg.inv(self.covariance)),
                                np.array([rt_diff[i], mz_diff[i]]).T)
                                for i in
                                range(len(rt_diff))])
                            if min(diff_score) < self.max_match_score:
                                self.possible_anchors[
                                    diff_score.argmin()].add_ms1_scan(
                                    scan.mzs[i], scan.rts[i],
                                    scan.sample_id,
                                    scan.peak_ids[i],
                                    scan.peak_statuses[i],
                                    est_drift, est_drift_sd)
                                self.matched_anchors.append(
                                    self.possible_anchors[diff_score.argmin()])
                                self.possible_anchors.pop(diff_score.argmin())
                            else:
                                self.matched_anchors.append(
                                    SimplePeak(scan.mzs[i], scan.rts[i],
                                               scan.sample_id,
                                               scan.peak_ids[i],
                                               scan.peak_statuses[i],
                                               est_drift, est_drift_sd))
                        else:
                            # creates new anchor
                            self.matched_anchors.append(
                                SimplePeak(scan.mzs[i], scan.rts[i],
                                           scan.sample_id,
                                           scan.peak_ids[i],
                                           scan.peak_statuses[i], est_drift,
                                           est_drift_sd))
                else:
                    peak_ids = np.array([anchor.peak_ids[-1] for anchor in
                                         self.matched_anchors])
                    np.array(self.matched_anchors)[
                        np.where(peak_ids == scan.peak_ids[0])[0]][
                        0].add_ms2_scan(scan)

    # def match_scans(self, scans, est_drift, est_drift_sd):
    #     for scan in scans:
    #         if scan.ms_level == 1:
    #             for i in range(len(scan.mzs)):
    #                 if len(self.possible_anchors) > 0:
    #                     rt_diff = np.array([anchor.est_rt - scan.rts[i] for anchor in
    #                                         self.possible_anchors])
    #                     mz_diff = np.array([anchor.est_mz - scan.mzs[i] for anchor in
    #                                         self.possible_anchors])
    #                     diff_score = np.array(
    #                         [np.matmul(
    #                             np.matmul(np.array([rt_diff[i], mz_diff[i]]),
    #                                       np.linalg.inv(self.covariance)), np.array(
    #                                 [rt_diff[i], mz_diff[i]]).T) for i in range(len(rt_diff))])
    #                     if min(diff_score) < self.max_match_score:
    #                         self.possible_anchors[diff_score.argmin()].add_ms1_scan(
    #                             scan.mzs[i], scan.rts[i], scan.sample_id, scan.peak_ids[i],
    #                             scan.peak_statuses[i], est_drift, est_drift_sd)
    #                         self.matched_anchors.append(self.possible_anchors[diff_score.argmin()])
    #                         self.possible_anchors.pop(diff_score.argmin())
    #                     else:
    #                         self.matched_anchors.append(
    #                             SimplePeak(scan.mzs[i], scan.rts[i], scan.sample_id,
    #                                        scan.peak_ids[i], scan.peak_statuses[i],
    #                                        est_drift, est_drift_sd))
    #                 else:
    #                     # creates new anchor
    #                     self.matched_anchors.append(SimplePeak(scan.mzs[i], scan.rts[i],
    #                                                            scan.sample_id, scan.peak_ids[i],
    #                                                            scan.peak_statuses[i], est_drift,
    #                                                            est_drift_sd))
    #         elif scan.ms_level == 2:
    #             # Note - In this system, all ms1s are assigned to anchors before any ms2 scans
    #             # take place related to that ms1 scan. When an anchor comes in, there is a
    #             # possibility that the ms1 scan gets moved from one anchor to another
    #             which_current_anchor = self.find_matched_anchor(scan)
    #             current_anchor = self.matched_anchors[which_current_anchor]
    #             possible_anchors_spectra = [anchor.spectra for anchor in self.possible_anchors]
    #             matched_anchors_spectra = [anchor.spectra for anchor in self.matched_anchors]
    #             possible_which_anchor = np.where(
    #                 np.array(possible_anchors_spectra) == current_anchor.spectra)[0]
    #             matched_which_anchor = np.where(
    #                 np.array(matched_anchors_spectra) == current_anchor.spectra)[0]
    #             if len(possible_which_anchor) == len(matched_which_anchor) == 0:
    #                 possible_or_matched = None
    #                 which_other_anchor = None
    #                 other_anchor = None
    #             elif len(possible_which_anchor) > 0:
    #                 possible_or_matched = 'possible'
    #                 which_other_anchor = possible_which_anchor
    #                 other_anchor = self.possible_anchors[possible_which_anchor]
    #             else:
    #                 possible_or_matched = 'matched'
    #                 which_other_anchor = matched_which_anchor
    #                 other_anchor = self.matched_anchors[possible_which_anchor]
    #
    #             if other_anchor is None:
    #                 # there are no matches at all, including to its currently asigned anchor
    #                 if scan.frag_spectra == current_anchor.spectra or \
    #                         current_anchor.spectra is None:
    #                     # add spectra to current anchor
    #                     self.matched_anchors[which_current_anchor].add_new_ms2_scan(scan)
    #                 else:
    #                     # create new spectra
    #                     self.matched_anchors.append(
    #                         SimplePeak(current_anchor.mzs[-1],
    #                                    current_anchor.rts[-1],
    #                                    current_anchor.sample_ids[-1],
    #                                    current_anchor.peak_ids[-1],
    #                                    current_anchor.peak_statuses[-1],
    #                                    current_anchor.est_drifts[-1],
    #                                    current_anchor.est_drifts_sd[-1]))
    #                     self.matched_anchors[which_current_anchor].remove_last_entry()
    #                     self.matched_anchors[-1].add_new_ms2_scan(scan)
    #             else:
    #                 # match found with another anchor
    #                 if possible_or_matched == 'possible':
    #                     self.possible_anchors[which_other_anchor].add_ms1_scan(
    #                         current_anchor.mzs[-1], current_anchor.rts[-1],
    #                         current_anchor.sample_ids[-1], current_anchor.peak_ids[-1],
    #                         current_anchor.peak_statuses[-1], current_anchor.est_drifts[-1],
    #                         current_anchor.est_drifts_sd[-1])
    #                     self.possible_anchors[which_other_anchor].add_new_ms2_scan(scan)
    #                 else:
    #                     self.matched_anchors[which_other_anchor].add_ms1_scan(
    #                         current_anchor.mzs[-1], current_anchor.rts[-1],
    #                         current_anchor.sample_ids[-1], current_anchor.peak_ids[-1],
    #                         current_anchor.peak_statuses[-1], current_anchor.est_drifts[-1],
    #                         current_anchor.est_drifts_sd[-1])
    #                     self.matched_anchors[which_other_anchor].add_new_ms2_scan(scan)
    #                 self.matched_anchors[which_current_anchor].remove_last_entry()
    #                 # TODO: what happens if anchor already has scan for that sample
    #                 # TODO: look at moving the element to an alternative ancho
    #                 # TODO: potentially add possibility of moving other elements
    #                 #  to other / new anchor

    def find_matched_anchor(self, ms2_scan):
        # returns the location of the anchor related to the ms2 scan
        sample_ids = np.array(
            [anchor.sample_ids[-1] for anchor in self.matched_anchors])
        peak_ids = np.array(
            [anchor.sample_ids[-1] for anchor in self.matched_anchors])
        anchor_where = np.where(
            sample_ids == ms2_scan.sample_id and peak_ids == ms2_scan.precursor_id)[
            0]
        return anchor_where


class SimpleDriftExperiment():
    def __init__(self, datasets, drift_model, rt_range, covariance,
                 max_match_score, sensitivity=1, specificity=1,
                 frag_method='random', N=0):
        # drift_model is a GP specified in notebook
        # frag_method - None gives full scan, 'random' gives random top N, 'targeted'
        # gives targeted top N
        self.datasets = datasets
        self.drift_model = drift_model
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.frag_method = frag_method
        self.N = N
        self.saved_datasets = [[] for d in datasets]
        self.peak_matching = SimplePeakMatching(covariance,
                                                max_match_score)  # initialise peak matching
        keys = list(self.datasets.keys())
        for idx in range(len(keys)):
            scan_time = rt_range[0]
            est_drift = 0
            est_drift_sd = 0
            while scan_time < rt_range[1]:
                new_scan_time = scan_time + 1
                scans = self.scan(scan_time, new_scan_time, keys[idx])
                saved_datasets = self.peak_matching.add_new_scans(scans,
                                                                  est_drift,
                                                                  est_drift_sd)
                self.saved_datasets[idx].append(saved_datasets)
                if idx > 0:
                    self.model_fit()
                    est_drift, est_drift_sd = self.model_predict(
                        new_scan_time + 1)
                scan_time = new_scan_time
            self.peak_matching.reset_current_statuses()

    def gp_predict(self, sample, scan_idx):
        self.drift_model.fit()
        # then basically reproduce plot in notebook
        NotImplementedError()

    def scan(self, initial_scan_time, new_scan_time, dataset_key):
        scans = []
        # do ms1 scan
        ms1_scan = SimpleScan(initial_scan_time, new_scan_time, dataset_key, 1)
        for p in self.datasets[dataset_key]:
            if initial_scan_time < p.rt < new_scan_time:
                if chem_test(p, self.sensitivity, self.specificity):
                    ms1_scan.add(p)
        scans.append(ms1_scan)
        # do ms2 scans
        n = min(self.N, len(ms1_scan.rts))
        if n != 0 and len(ms1_scan.rts) != 0:
            if self.frag_method == 'random':
                which_chems = random.sample(ms1_scan.peak_ids, n)
                for p in self.datasets[dataset_key]:
                    if p.id in which_chems:
                        ms2_scan = SimpleScan(initial_scan_time, new_scan_time,
                                              dataset_key, 2)
                        ms2_scan.add(p)
                        scans.append(ms2_scan)
            else:
                print('Incorrect frag method specified')
        return scans

    def model_fit(self):
        self.peak_matching.update_xy()
        if len(self.peak_matching.X) > 0:
            self.drift_model.fit(self.peak_matching.X, self.peak_matching.y)

    def model_predict(self, current_rt):
        if len(self.peak_matching.X) > 0:
            mean, sd = self.drift_model.predict(np.array([[current_rt]]),
                                                return_std=True)
        else:
            mean = 0
            sd = 0  # TODO: fix this properly
        return mean, sd


def chem_test(chem, sensitivity, specificity):
    """
    sensitivity - probability of correctly identifying a peak
    specificity - probability of correctly identifying noise
    """
    if chem.peak_status:
        result = np.random.binomial(1, sensitivity, 1)
    else:
        result = np.random.binomial(1, 1 - specificity, 1)
    return (result == 1)[0]


class SimpleMatchingScore():
    def __init__(self, simple_experiment):
        """
        definition: The maximal group for a chemical is the group of peaks which contains the
        highest number of that chemical. i.e. if group 1 = {chem 1, chem 1, chem 1, some other
        chems} and group 2 = {chem 1, some other chems} then group 1 is the maximal group
        for chem 1

        correct matching
            - sum of the number of chemicals in their maximal group, where the maximal group
            only contains that chemical
        polluted matching
            - sum of the number of chemicals in their maximal group, where the maximal group
            contains other chemicals / noise
        incorrect matching
            - sum of all the chemicals not in the maximal matched group that are in a group
            with another chemical / noise
        split matching
            - sum of all the chemicals not in the maximal matched group that are in a group
            with just chemicals of the same type

        non-peaks separated
            - non peaks that are in a group by themselves
        non-peaks contaminating
            - non peaks that are in a group with chemicals
        non-peaks connected
            - non peaks that get matched together incorrectly

        """
        self.simple_experiment = simple_experiment
        # results for chemicals
        self.correct_matching = 0
        self.polluted_matching = 0
        self.incorrect_matching = 0
        self.split_matching = 0

        # results for noise
        self.noise_separated = 0
        self.noise_contaminating = 0
        self.noise_connected = 0

        # Find chemical IDs
        chem_ids, noise_ids = self._get_all_ids()

        # Find maximal groups
        self.maximal_groups = self._get_maximal_groups(chem_ids)

        # Get chem scores
        for idx in range(len(chem_ids)):
            mg = self.maximal_groups[idx]
            chem_idx = chem_ids[idx]
            self._get_chem_scores(mg, chem_idx)

        # Get noise scores
        for noise_id in noise_ids:
            self._get_noise_scores(noise_id)

        # calculate percentage scores for chems
        total_observed_chems = self.correct_matching + self.polluted_matching + \
            self.incorrect_matching + self.split_matching
        self.correct_matching_percentage = self.correct_matching / total_observed_chems
        self.polluted_matching_percentage = self.polluted_matching / total_observed_chems
        self.incorrect_matching_percentage = self.incorrect_matching / total_observed_chems
        self.split_matching_percentage = self.split_matching / total_observed_chems

        # calculate percentage scores for noise
        total_observed_noise = self.noise_separated + \
            self.noise_contaminating + self.noise_connected
        if total_observed_noise > 0:
            self.noise_separated_percentage = self.noise_separated / total_observed_noise
            self.noise_contaminating_percentage = self.noise_contaminating / total_observed_noise
            self.noise_connected_percentage = self.noise_separated / total_observed_noise

    def _get_all_ids(self):
        chem_ids = []
        noise_ids = []
        keys = list(self.simple_experiment.datasets.keys())
        for key in keys:
            for chem in self.simple_experiment.datasets[key]:
                if chem.peak_status:
                    chem_ids.append(chem.id)
                else:
                    noise_ids.append(chem.id)
        chem_ids = list(np.unique(np.array(chem_ids)))
        noise_ids = list(np.unique(np.array(noise_ids)))
        return chem_ids, noise_ids

    def _get_maximal_groups(self, chem_ids):
        maximal_groups = []
        for idx in chem_ids:
            anchors = self.simple_experiment.peak_matching.anchors
            group_total = np.array(
                [sum(np.array(a.peak_ids) == idx) for a in anchors])
            if sum(group_total == group_total.max()) == 1:
                maximal_groups.append(group_total.argmax())
            else:
                group_options = np.where(group_total == group_total.max())[0]
                group_size = np.array(
                    [len(anchors[op].peak_ids) for op in group_options])
                maximal_groups.append(group_options[group_size.argmin()])
        return maximal_groups

    def _get_chem_scores(self, mg, chem_idx):
        group_total = np.array([len(group.peak_ids) for group in
                                self.simple_experiment.peak_matching.anchors])
        group_chem_total = np.array(
            [sum(np.array(group.peak_ids) == chem_idx) for group in
             self.simple_experiment.peak_matching.anchors])
        if group_total[mg] == group_chem_total[mg]:
            self.correct_matching += group_total[mg]
            self.polluted_matching += 0
        else:
            self.correct_matching += 0
            self.polluted_matching += group_total[mg]
        group_total = np.delete(group_total, mg)
        group_chem_total = np.delete(group_chem_total, mg)
        for i in range(len(group_chem_total)):
            if group_chem_total[i] == group_total[i]:
                self.split_matching += group_chem_total[i]
            else:
                self.incorrect_matching += group_chem_total[i]

    def _get_noise_scores(self, noise_id):
        group_total = np.array([len(group.peak_ids) for group in
                                self.simple_experiment.peak_matching.anchors])
        group_noise_total = np.array(
            [sum(np.array(group.peak_ids) == noise_id) for group in
             self.simple_experiment.peak_matching.anchors])
        for i in range(len(group_noise_total)):
            if group_noise_total[i] > 0:
                if group_noise_total[i] == group_total[i] == 1:
                    self.noise_separated += 1
                else:
                    peak_statuses = \
                        self.simple_experiment.peak_matching.anchors[
                            i].peak_statuses
                    if all(np.array(peak_statuses) == False): # noqa
                        self.noise_connected += 1
                    else:
                        self.noise_contaminating += 1


###################################################################################################
# ViMMS Drift Method Classes
###################################################################################################


class VimmsPeakMatching(PeakMatching):
    def __init__(self, covariance, max_match_score):
        super().__init__(covariance, max_match_score)
