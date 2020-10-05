import matplotlib.pyplot as plt
import numpy as np
import scipy
from events import Events
from scipy.stats import skewnorm

from vimms.Common import POSITIVE, ScanParameters
from vimms.MassSpec import IndependentMassSpectrometer, Scan


class Person(object):
    def __init__(self, person_id, age, comorbidities, covid_status=False, covid_level=0, covid_immunity=False,
                 viral_disease_status=False, viral_disease_level=0):
        self.id = person_id
        self.age = age
        self.comorbidities = comorbidities
        self.covid_status = [covid_status]
        self.covid_level = [covid_level]
        self.covid_immunity = [covid_immunity]
        self.viral_disease_status = [viral_disease_status]
        self.viral_disease_level = [viral_disease_level]
        self.symptoms_level = [symptoms_model(covid_level, viral_disease_level)]
        self.time_measurements = [0]
        self.test_results = [None]
        self.alive = [True]
        self.covid = None
        self.viral_disease = None
        self.observed_times = [0]
        self.observed_symptoms = [self.symptoms_level[0]]
        self.observed_viral_disease_status = [viral_disease_status]
        self.observed_covid_status = [covid_status]

    def update(self, t, test_params, covid_params, viral_params, do_test=False, ms_level=1):
        self.time_measurements.append(t)
        self._update_covid(covid_params)
        self._update_viral_disease(viral_params)
        symptoms = symptoms_model(self.covid_level[-1], self.viral_disease_level[-1])
        self.symptoms_level.append(symptoms)
        if ms_level == 2 and do_test:
            test_result = covid_test(self.covid_status[-1], self.covid_level[-1], test_params['sensitivity'],
                                     test_params['specificity'])
            self.test_results.append(test_result)
        else:
            self.test_results.append(None)
        if ms_level == 1:
            self.observed_times.append(t)
            self.observed_symptoms.append(symptoms)
            self.observed_viral_disease_status.append(self.viral_disease_status[-1])
            self.observed_covid_status.append(self.covid_status[-1])

    def _update_covid(self, covid_params):
        if self.covid_immunity[-1]:
            # people who already have immunity continue to be immune, have no status and not have covid
            self.covid_status.append(False)
            self.covid_level.append(0)
            self.covid_immunity.append(True)
        else:
            if self.covid_status[-1]:
                # progression of people with covid
                self.covid_level.append(self.covid.step())
                if self.covid_level[-1] == 0:
                    self.covid_immunity.append(True)
                    self.covid_status.append(False)
                else:
                    self.covid_immunity.append(False)
                    self.covid_status.append(True)
            else:
                get_covid = np.random.binomial(1, covid_params['prevelance'], 1)[0]
                if get_covid == 1:
                    self.covid = Disease(covid_params['poisson_param'], covid_params['uniform_min'],
                                         covid_params['uniform_max'], covid_params['noise_sd'])
                    self.covid_level.append(self.covid.step())
                    self.covid_status.append(True)
                else:
                    self.covid_level.append(0)
                    self.covid_status.append(False)
                self.covid_immunity.append(False)

    def _update_viral_disease(self, viral_params):
        if self.covid_status[-1]:
            # if have covid, then dont have cold
            self.viral_disease_level.append(0)
            self.viral_disease_status.append(False)
        else:
            # if do not have covid, then are able to get a disease
            if self.viral_disease_status[-1]:
                # progression of people with viral disease
                self.viral_disease_level.append(self.viral_disease.step())
                if self.viral_disease_level[-1] == 0:
                    self.viral_disease_status.append(False)
                else:
                    self.viral_disease_status.append(True)
            else:
                get_viral = np.random.binomial(1, viral_params['prevelance'], 1)[0]
                if get_viral == 1:
                    self.viral_disease = Disease(viral_params['poisson_param'], viral_params['uniform_min'],
                                                 viral_params['uniform_max'], viral_params['noise_sd'])
                    self.viral_disease_level.append(self.viral_disease.step())
                    self.viral_disease_status.append(True)
                else:
                    self.viral_disease_level.append(0)
                    self.viral_disease_status.append(False)

    def plot_symptoms(self):
        plt.figure(figsize=(15, 6))
        for i in range(len(self.observed_times)):
            time = self.observed_times[i]
            symptoms = self.observed_symptoms[i]
            if self.observed_viral_disease_status[i]:
                marker = 'x'
            else:
                marker = 'o'
            if self.observed_covid_status[i]:
                colour = 'red'
            else:
                colour = 'green'
            plt.scatter(time, symptoms, c=colour, marker=marker)
        for i in range(len(self.test_results)):
            test_result = self.test_results[i]
            time = self.time_measurements[i]
            if test_result is not None:
                if test_result == 1:
                    test_colour = 'red'
                else:
                    test_colour = 'green'
                plt.axvline(x=time, c=test_colour)
        # TODO: add a legend
        plt.show()


def create_population(N):
    population = []
    ages = sample_person_ages(N)
    comorb = sample_person_commorbidities(N, ages)
    for i in range(N):
        person = Person(i, ages[i], comorb[i])
        population.append(person)
    return population


def sample_person_ages(N):
    ages = skewnorm.rvs(4, size=N)
    ages = ages - min(ages)
    ages = np.round(ages * 30)
    return ages


def sample_person_commorbidities(N, ages):
    comorbs = np.random.poisson(ages / 50, N)
    return comorbs


def symptoms_model(covid_level, viral_disease_level):
    symptoms_level = max(covid_level, viral_disease_level)
    return max(symptoms_level, 0)


def covid_test(covid_status, covid_symptoms, test_sensitivity, test_specificity):
    if covid_status:
        test_status = np.random.binomial(1, test_sensitivity, 1)
    else:
        test_status = np.random.binomial(1, 1 - test_sensitivity, 1)
    return test_status[0]


class Disease(object):
    def __init__(self, poisson_param, uniform_min, uniform_max, noise_sd):
        t = np.array(range(max(1, np.random.poisson(poisson_param, 1)[0])))
        daily_increase = np.random.uniform(uniform_min, uniform_max, 1)
        level = daily_increase * np.minimum(t, max(t) - t) + np.random.normal(0, noise_sd, len(t))
        self.level = np.maximum(0, level).tolist()

    def step(self):
        try:
            disease_level = self.level.pop(0)
        except:
            disease_level = 0
        return disease_level


class Disease2(object):
    def __init__(self, start_time, disease_params):
        mean = np.random.uniform(disease_params['mean_min'], disease_params['mean_max'])
        sd = np.random.uniform(disease_params['sd_min'], disease_params['sd_max'])
        self.model = scipy.stats.norm(mean + start_time, 1)
        self.scale = disease_params['scale']
        self.min_intensity = disease_params['min_intensity']

    def disease_intensity(self, current_time):
        intensity = self.scale * self.model.pdf(current_time)
        if intensity > self.min_intensity:
            return intensity
        else:
            return 0


class PopulationModel(object):
    def __init__(self, N, test_params, covid_params, viral_params, covid_model):
        self.population = create_population(N)
        self.test_params = test_params
        self.covid_params = covid_params
        self.viral_params = viral_params
        self.covid_model = covid_model

    def update(self, t, params):
        ms_level = params.get(ScanParameters.MS_LEVEL)
        self.update_covid_model(t)
        tests = self.choose_tests(params)
        self.update_population(t, tests, ms_level)

    def update_covid_model(self, t):
        return None

    def update_population(self, t, tests, ms_level):
        for i in range(len(self.population)):
            self.population[i].update(t, self.test_params, self.covid_params, self.viral_params, tests[i], ms_level)

    def choose_tests(self, params):
        ms_level = params.get(ScanParameters.MS_LEVEL)
        tests = np.array([False for i in self.population])
        if ms_level == 2:
            precursor_mz = params.get(ScanParameters.PRECURSOR_MZ).precursor_mz
            tests[precursor_mz] = True
        return tests.tolist()

    def plot_person_symptoms(self, id):
        self.population[id].plot_symptoms()  # TODO: change to use ID


class CovidMassSpectrometer(IndependentMassSpectrometer):
    def __init__(self, population_model):
        # current scan index and internal time
        self.idx = 0
        self.time = 0

        # current task queue
        self.processing_queue = []
        self.environment = None

        self.events = Events((self.MS_SCAN_ARRIVED, self.ACQUISITION_STREAM_OPENING, self.ACQUISITION_STREAM_CLOSED,
                              self.STATE_CHANGED,))
        self.event_dict = {
            self.MS_SCAN_ARRIVED: self.events.MsScanArrived,
            self.ACQUISITION_STREAM_OPENING: self.events.AcquisitionStreamOpening,
            self.ACQUISITION_STREAM_CLOSED: self.events.AcquisitionStreamClosing,
            self.STATE_CHANGED: self.events.StateChanged
        }

        # the list of all chemicals in the dataset
        self.chemicals = population_model.population
        self.population_model = population_model
        self.ionisation_mode = POSITIVE  # currently unused

        # stores the chromatograms start and end rt for quick retrieval
        self.chrom_min_rts = np.array([0 for chem in self.chemicals])
        self.chrom_max_rts = np.array([1000000 for chem in self.chemicals])

        # here's where we store all the stuff to sample from
        self.peak_sampler = None

        # required to sample for different scan durations based on (N, DEW) in the hybrid controller
        self.current_N = 0
        self.current_DEW = 0

        self.add_noise = False  # whether to add noise to the generated fragment peaks
        self.fragmentation_events = []  # which chemicals produce which peaks

        self.isolation_transition_window = 'rectangular'
        self.isolation_transition_window_params = None

        self.scan_duration_dict = None

    def _sample_scan_duration(self, current_DEW, current_N, current_level, next_level):
        current_scan_duration = 1 / (current_N + 1)
        return current_scan_duration

    def _get_scan(self, scan_time, params):
        ms_level = params.get(ScanParameters.MS_LEVEL)
        scan_id = self.idx
        # update population
        self.population_model.update(scan_time, params)
        # get results for ms1 scan
        if ms_level == 1:
            scan_mzs = np.array([p.id for p in self.population_model.population])
            scan_intensities = np.array([p.symptoms_level[-1] for p in self.population_model.population])
            scan = Scan(scan_id, scan_mzs, scan_intensities, ms_level, scan_time, scan_duration=None,
                        scan_params=params)
        # get results for ms2 scan
        if ms_level == 2:
            scan = Scan(scan_id, np.array([100]), np.array([100]), ms_level, scan_time, scan_duration=None,
                        scan_params=params)
        return scan
