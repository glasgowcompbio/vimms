import random
from abc import abstractmethod

import numpy as np
import GPy

from mass_spec_utils.library_matching.spectral_scoring_functions import cosine_similarity
from mass_spec_utils.library_matching.spectrum import Spectrum

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
            self.chem_rts_by_injection = [
                [sample[i] for i in idxes] for sample in chem_rts_by_injection
            ]
        else:
            self.chem_rts_by_injection = chem_rts_by_injection
        self.chem_to_idx = {
            chem if chem.base_chemical is None else chem.base_chemical : idx 
            for chem, idx in zip(chemicals, range(len(chem_rts_by_injection[0])))
        }
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])
        self.available = [False] * len(self.chem_rts_by_injection[0])
        self.mode = OraclePointMatcher.MODE_FRAGPAIRS if mode is None else mode

    def _next_model(self):
        self.not_sent = [True] * len(self.chem_rts_by_injection[0])

    def send_training_data(self, model, scan, roi, inj_num):
        if (self.mode == OraclePointMatcher.MODE_FRAGPAIRS):
            if (not scan.fragevent is None):

                for fe in scan.fragevent:
                    parent_chem = (
                        fe.chem
                        if fe.chem.base_chemical is None
                        else fe.chem.base_chemical
                    )

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

            for i, (y, x) in enumerate(zip(self.chem_rts_by_injection[inj_num], 
                                            self.chem_rts_by_injection[0])
                                       ):
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
                    current_score, _ = cosine_similarity(spectrum, 
                                                         s, 
                                                         self.mass_tol, 
                                                         self.min_match
                                                        )
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
                self.model = GPy.models.GPRegression(
                    np.array(Y).reshape((len(Y), 1)), 
                    np.array(X).reshape((len(X), 1)),
                    kernel=self.kernel
                )
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