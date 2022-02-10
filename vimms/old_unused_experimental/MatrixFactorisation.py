# flake8: noqa

import bisect
import copy
import math

import numpy as np
import pylab as plt
import scipy
from loguru import logger


class BlockData(object):
    def __init__(self, datasets, mz_step, rt_step, rt_range=[(0, 1450)], mz_range=[(50, 1070)]):
        self.datasets = datasets
        self.mz_step = mz_step
        self.rt_step = rt_step
        self.rt_range = rt_range
        self.mz_range = mz_range
        self.keys = []
        for j in range(len(self.datasets)):
            self.keys.append(list(self.datasets[j].file_spectra.keys()))
        self.mz_bin_lower = np.arange(mz_range[0][0], mz_range[0][1], mz_step)
        self.rt_bin_lower = np.arange(rt_range[0][0], rt_range[0][1], rt_step)
        self.n_mz_bin_lower = len(self.mz_bin_lower)
        self.n_rt_bin_lower = len(self.rt_bin_lower)

        self.intensity_mats = []
        for j in range(len(self.datasets)):
            for i in range(len(self.keys[j])):
                self.intensity_mats.append(self._block_file(i, j))
                logger.debug("Processed " + self.keys[j][i])

    def _block_file(self, num, data_num):
        intensity_mat = np.zeros((self.n_mz_bin_lower, self.n_rt_bin_lower), np.double)
        spectra = self.datasets[data_num].file_spectra[self.keys[data_num][num]]
        c1 = 0
        for scan_num in spectra:
            scan_rt = spectra[scan_num].scan_time[0]
            if scan_rt < self.rt_bin_lower[0]:
                continue
            if scan_rt > self.rt_bin_lower[-1] + self.rt_step:
                continue
            else:
                rt_pos = bisect.bisect_right(self.rt_bin_lower, scan_rt)
                rt_pos -= 1
                for peak_num in range(len(spectra[scan_num].mz)):
                    mz = spectra[scan_num].mz[peak_num]
                    intensity = spectra[scan_num].i[peak_num]
                    if mz < self.mz_bin_lower[0]:
                        continue
                    if mz > self.mz_bin_lower[-1] + self.mz_step:
                        break
                    mz_pos = bisect.bisect_right(self.mz_bin_lower, mz)
                    mz_pos -= 1

                    intensity_mat[mz_pos, rt_pos] += intensity
        return intensity_mat

    def plot(self, data_num):
        logger.warning("Warning: Python prints plots in a stupid stupid way!")
        plt.imshow(np.log(self.intensity_mats[data_num][-1:0:-1] + 1), aspect='auto')

    def combine(self, plot=True):
        combined = []
        for mat in self.intensity_mats:
            combined.append(list(mat.flatten()))
        return combined


def gibbs_sampler(X, observed, R, prior_u, prec_u, prior_v, prec_v, alpha, n_its=1000, burn_in=100, true_V=[],
                  sample_known=True):
    # initialise
    N, M = X.shape
    U = np.random.normal(size=(N, R))
    if len(true_V) == 0:
        V = np.random.normal(size=(M, R))
    else:
        V = true_V
    tot_U = np.zeros((N, R))
    tot_V = np.zeros((M, R))
    samples_U = []
    samples_V = []
    all_err = []
    range_U = range(N)
    if sample_known is False:
        range_U = np.where(np.sum(observed, axis=1) != len(observed[0, :]))[0].tolist()
    for it in range(n_its):
        # loop over u, updating them
        # first compute the covariance - shared if all data observed
        prec_mat = prec_u + alpha * np.dot(V.T, V)
        cov_mat = np.linalg.inv(prec_mat)
        for n in range_U:
            if observed[n, :].sum() < M:
                # not all data observed, compute specific precision
                this_prec_mat = prec_u + alpha * np.dot(np.dot(V.T, np.diag(observed[n, :])), V)
                this_cov_mat = np.linalg.inv(this_prec_mat)
            else:
                this_prec_mat = prec_mat
                this_cov_mat = cov_mat
            s = np.zeros(R)
            for m in range(M):
                if observed[n, m]:
                    s += X[n, m] * V[m, :]
            s *= alpha
            s += np.dot(prec_u, prior_u)
            cond_mu = np.dot(this_cov_mat, s)
            U[n, :] = np.random.multivariate_normal(cond_mu, this_cov_mat)

        # loop over v updating them
        # first covariance
        if len(true_V) == 0:
            prec_mat = prec_v + alpha * np.dot(U.T, U)
            cov_mat = np.linalg.inv(prec_mat)
            for m in range(M):
                if observed[:, m].sum() < N:
                    this_prec_mat = prec_v + alpha * np.dot(np.dot(U.T, np.diag(observed[:, m])), U)
                    this_cov_mat = np.linalg.inv(this_prec_mat)
                else:
                    this_prec_mat = prec_mat
                    this_cov_mat = cov_mat

                s = np.zeros(R)
                for n in range(N):
                    if observed[n, m]:
                        s += X[n, m] * U[n, :]
                s *= alpha
                s += np.dot(prec_v, prior_v)
                cond_mu = np.dot(this_cov_mat, s)
                V[m, :] = np.random.multivariate_normal(cond_mu, this_cov_mat)
        if it > burn_in:
            tot_U += U
            tot_V += V
            samples_U.append(copy.deepcopy(U))
            samples_V.append(copy.deepcopy(V))
        recon_error = np.sqrt(((X - np.dot(U, V.T)) ** 2).mean())
        all_err.append(recon_error)
    if len(true_V) == 0:
        return tot_U / (n_its - burn_in), tot_V / (n_its - burn_in)
    else:
        if sample_known is True:
            return samples_U
        else:
            updated_samples_U = []
            for i in range(len(samples_U)):
                updated_samples_U.append(samples_U[i][range_U, :])
            return range_U, updated_samples_U


class VB_PCA(object):
    def __init__(self, Y, Z, D, MaxIts=100, a=1, b=1, tol=1e-3, compute_LB=False, VB_PCA_model=None):

        # intialise parameters
        self.Y = Y
        self.Z = Z
        self.D = D
        ZY = Z * Y
        self.B = []
        self.e_tau = []
        e_tau = a / b
        self.e_tau.append(e_tau)
        self.e_log_tau = [np.log(e_tau)]
        self.N = Y.shape[0]
        self.M = Y.shape[1]
        self.e_w = np.random.normal(0, 1, (self.M, self.D))
        self.e_X = np.random.normal(0, 1, (self.N, self.D))
        self.e_wwt = []
        self.e_XXt = []
        for m in range(self.M):
            self.e_wwt.append(np.identity(self.D) + np.matmul(self.e_w[m, :][np.newaxis].T, self.e_w[m, :][np.newaxis]))
        for n in range(self.N):
            self.e_XXt.append(np.identity(self.D) + np.matmul(self.e_X[n, :][np.newaxis].T, self.e_X[n, :][np.newaxis]))
        self.sigx = [[] for i in range(self.N)]
        self.sigw = [[] for i in range(self.M)]
        # run code
        for it in range(MaxIts):

            # update X
            for n in range(self.N):
                Zlist = [vec * np.ones((self.D, self.D)) for vec in Z[n, :]]
                self.sigx[n] = np.linalg.inv(np.identity(self.D) + e_tau * sum(np.array(self.e_wwt) * np.array(Zlist)))
                self.e_X[n, :] = e_tau * np.matmul(self.sigx[n], np.sum(
                    np.multiply(self.e_w, np.array([(ZY[n, :]).tolist() for i in range(self.D)]).T), axis=0))
                self.e_XXt[n] = self.sigx[n] + np.matmul(self.e_X[n, :][np.newaxis].T, self.e_X[n, :][np.newaxis])

            # update W
            for m in range(self.M):
                Zlist = [vec * np.ones((self.D, self.D)) for vec in Z[:, m]]
                self.sigw[m] = np.linalg.inv(np.identity(self.D) + e_tau * sum(np.array(self.e_XXt) * np.array(Zlist)))
                self.e_w[m, :] = e_tau * np.matmul(self.sigw[m], np.sum(
                    np.multiply(self.e_X, np.array([(Y[:, m] * Z[:, m]).tolist() for i in range(self.D)]).T), axis=0))
                self.e_wwt[m] = self.sigw[m] + np.matmul(self.e_w[m, :][np.newaxis].T, self.e_w[m, :][np.newaxis])

            # update tau
            e = a + sum(sum(Z)) / 2
            outer_expect = 0
            RSS = 0
            for n in range(self.N):
                for m in range(self.M):
                    outer_expect += Z[n, m] * (np.trace(np.matmul(self.e_wwt[m], self.sigx[n])) + np.matmul(
                        np.matmul(self.e_X[n, :], self.e_wwt[m]), self.e_X[n, :][np.newaxis].T))
                    RSS += (ZY[n, m] ** 2) - 2 * np.matmul(self.e_w[m].T, self.e_X[n]) * (ZY[n, m])
            f = b + 0.5 * RSS + 0.5 * outer_expect
            e_tau = e / f
            self.e_tau.append(e_tau[0])
            e_log_tau = np.mean(np.log(np.random.gamma(shape=e, scale=1 / f, size=1000)))
            self.e_log_tau.append(e_log_tau)

            # Compute the bound
            if compute_LB is True:
                LB = a * np.log(b) + (a - 1) * e_log_tau - b * e_tau - scipy.special.loggamma(a)
                LB -= (e * np.log(f) + (e - 1) * e_log_tau - f * e_tau - scipy.special.loggamma(e))

                for n in range(self.N):
                    LB += (-(self.D / 2) * np.log(2 * math.pi) - 0.5 * sum(np.diag(self.sigx[n])) + sum(
                        self.e_X[n, :] ** 2))
                    LB -= (-(self.D / 2) * np.log(2 * math.pi) - 0.5 * np.log(
                        np.linalg.det(self.sigx[n])) - 0.5 * self.D)

                for m in range(self.M):
                    LB += (-(self.D / 2) * np.log(2 * math.pi) - 0.5 * sum(np.diag(self.sigw[m])) + sum(
                        self.e_w[m, :] ** 2))
                    logger.debug((-(self.D / 2) * np.log(2 * math.pi) - 0.5 * np.log(
                        np.linalg.det(self.sigw[m])) - 0.5 * self.D))
                    LB -= (-(self.D / 2) * np.log(2 * math.pi) - 0.5 * np.log(
                        np.linalg.det(self.sigw[m])) - 0.5 * self.D)

                # likelihood bit
                LB += (-(self.N * self.M / 2) * np.log(2 * math.pi) + (
                        self.N * self.M / 2) * e_log_tau - 0.5 * e_tau * sum(
                    sum((ZY ** 2))) - 2 * sum(
                    sum(Z * (np.multiply(np.matmul(self.e_w, self.e_X.T).T, Y)))) + outer_expect)
                self.B.append(LB)

                # break if change in bound is less than the tolerance
                if it > 2:
                    if abs(self.B[-1] - self.B[-2]) < tol:
                        break
        # reconstruct Y
        self.Y_reconstructed = np.matmul(self.e_X, self.e_w.T)

    def update(self, new_Z, new_Y):
        # assume last Y is only new observations
        self.N = new_Y.shape[0]
        self.M = new_Y.shape[1]
        if self.Z.shape[0] < new_Z.shape[0]:
            self.e_X = np.concatenate((self.e_X, np.array([[0 for i in range(self.D)]])))
            self.sigx.append(np.identity(self.D))
        prec_x = np.identity(self.D) + np.dot(np.dot(self.e_w.T, np.diag(new_Z[-1, :])), self.e_w) * self.e_tau[-1]
        self.sigx[-1] = np.linalg.inv(prec_x)
        self.e_X[-1] = np.dot(self.sigx[-1], np.dot(self.e_w.T, (new_Y[-1,] * new_Z[-1,]))) * self.e_tau[-1]
        self.Y = new_Y
        self.Z = new_Z
        self.Y_reconstructed = np.matmul(self.e_X, self.e_w.T)
