import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from vimms.BOMAS import GetScaledValues

PARAM_RANGE_N0 = [[0, 250]]
PARAM_RANGE_N1 = [[0, 250], [0, 500], [0, 100], [1, 50]]


def MSmixture(theta, y, t, N):
    mean = np.array([theta[0] for i in y])
    if N == 1:
        mean += (theta[1] ** 2) * norm.pdf(t, abs(theta[2]), abs(theta[3]))
    return sum((y - mean) ** 2)


def Minimise_MSmixture(y, t, N, param_range_init, method='Nelder-Mead', restarts=10):
    init_values = GetScaledValues(restarts, param_range_init)
    opt_values = []
    opt_mins = []
    for i in range(restarts):
        # model_results = minimize(MSmixture, init_values[:, i], args=(y, t, N), method=method)
        model_results = minimize(MSmixture_posterior, init_values[:, i], args=(y, t, N), method=method)
        opt_mins.append(model_results)
        opt_values.append(model_results['fun'])
    final_model = opt_mins[np.array(opt_values).argmin()]
    min_value = np.array(opt_values).min()
    return final_model['x'], min_value


def GetPlot_MSmixture(t, theta, N):
    prediction = np.array([float(theta[0]) for i in t])
    if N == 1:
        prediction += np.array(theta[1] * norm.pdf(t, theta[2], theta[3]))
    return t, prediction


def MSmixture_posterior(theta, y, t, N, sigma=None, prior_mu=None, prior_var=None, neg_like=True):
    mean = np.array([theta[0] for i in y])
    if N == 1:
        mean += (theta[1] ** 2) * norm.pdf(t, abs(theta[2]), abs(theta[3]))
    var = sum((y - mean) ** 2) / (len(y))
    log_like = sum(np.log(norm.pdf(y, mean, var)))
    if neg_like:
        return -log_like
    return log_like


class SMC_MSmixture(object):
    def __init__(self, n_particles, n_mixtures, prior_mu, prior_var, jitter_params, prior_sigsq=None):
        self.n_particles = n_particles
        self.n_mixtures = n_mixtures
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        self.prior_sigsq = prior_sigsq
        self.jitter_params = jitter_params
        self.t = []
        self.y = []

        # get inital particles
        self.current_particles = np.random.multivariate_normal(prior_mu, np.diagflat(prior_var), self.n_particles)
        self.particles = []

    def update(self, new_t, new_y):
        # add new data to current data
        self.t.append(new_t)
        self.y.append(new_y)
        # get the weights
        self.current_weights = self._get_weights()
        # resample particles
        self.current_particles = self._get_resampled_particles()
        # add jitter
        self.current_particles = self._add_jitter()
        # update particles
        self.particles.append(self.current_particles)

    def _get_resampled_particles(self):
        updated_particles = self.current_particles[
            np.random.choice(self.n_particles, self.n_particles, p=self.current_weights)]
        return updated_particles

    def _add_jitter(self):
        noise = np.random.normal(loc=0, scale=self.jitter_params, size=self.current_particles.shape)
        return self.current_particles + noise

    def _get_weights(self):
        # get posteriors
        weights = [MSmixture_posterior(self.current_particles[i], self.y, self.t, self.n_mixtures, neg_like=False) for i
                   in
                   range(self.n_particles)]
        updated_weights = np.exp(np.array(weights) - np.array(weights).max())
        # re weight
        normalised_weights = np.exp(updated_weights) / sum(np.exp(updated_weights))
        return normalised_weights
