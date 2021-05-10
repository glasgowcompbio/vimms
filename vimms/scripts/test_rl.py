import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim

from vimms.Common import *
from vimms.Chemicals import ChemicalMixtureCreator, UniformRTAndIntensitySampler
from vimms.ChemicalSamplers import UniformMZFormulaSampler
from vimms.Evaluation import evaluate_simulated_env
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController, AgentBasedController
from vimms.Agent import TopNDEWAgent, ReinforceAgent, Pi, train
from vimms.Environment import Environment
from vimms.Noise import UniformSpikeNoise

base_dir = os.path.abspath(
    'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Trained Models')
hmdb = load_obj(Path(base_dir, 'hmdb_compounds.p'))

set_log_level_warning()

no_episodes = 20
n_chemicals = 100
rt_range = [(0, 400)]
min_rt = 0
max_rt = rt_range[0][1] + 50
min_mz = 100
max_mz = 500
noise_density = 0.3
noise_level = 1e5
early_stop = None

datasets = []
for i in range(no_episodes):
    if i % 10 == 0: print(i)

    # df = DatabaseFormulaSampler(hmdb, min_mz=min_mz, max_mz=max_mz)
    df = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
    ris = UniformRTAndIntensitySampler(min_rt=rt_range[0][0], max_rt=rt_range[0][1], max_log_intensity=np.log(1e10))

    cm = ChemicalMixtureCreator(df, rt_and_intensity_sampler=ris)
    chems = cm.sample(n_chemicals, 2, include_adducts_isotopes=False)
    datasets.append(chems)

isolation_window = 0.7
N = 10
rt_tol = 10
mz_tol = 10
min_ms1_intensity = 1e4
ionisation_mode = POSITIVE

out_dim = 10
in_dim = int(max_mz - min_mz)
hidden_dim = 32
gamma = 0.99
lr = 0.01

pbar = False
write_mzML = False


def run_experiment(datasets, controller_name, write_mzML=False, pbar=False, early_stop=None):
    # initial setup
    pi = None
    if controller_name == 'REINFORCE':
        pi = Pi(in_dim, out_dim, hidden_dim)
        optimizer = optim.Adam(pi.parameters(), lr=lr)

    # run episodes
    env_list = []
    losses = []
    total_rewards = []
    for i in range(len(datasets)):
        if early_stop is not None:
            if i >= early_stop:
                break

        if i % 10 == 0:
            print(i)
            write_mzML = True
        else:
            write_mzML = False

        if i == len(datasets) - 1:
            write_mzML = True

        if controller_name == 'TopN':
            controller = TopNController(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity,
                                        ms1_shift=0,
                                        initial_exclusion_list=None, force_N=False)

        elif controller_name == 'TopNAgent':
            agent = TopNDEWAgent(N, min_ms1_intensity, mz_tol, rt_tol)
            controller = AgentBasedController(ionisation_mode, agent)

        elif controller_name == 'REINFORCE':
            agent = ReinforceAgent(pi, N, min_ms1_intensity, mz_tol, rt_tol, min_mz, max_mz)
            controller = AgentBasedController(ionisation_mode, agent)

            # run the simulation
        out_dir = 'results' if write_mzML is True else None
        out_file = out_file = 'test_%s_%d.mzML' % (controller_name, i) if write_mzML is True else None

        spike_noise = UniformSpikeNoise(noise_density, noise_level, min_mz=min_mz, max_mz=max_mz)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None, spike_noise=spike_noise)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=pbar, out_dir=out_dir, out_file=out_file)
        env.run()
        env_list.append(env)

        # train REINFORCE model
        if pi is not None:
            loss = train(pi, optimizer, gamma)
            total_reward = sum(pi.rewards)
            positive_hits = np.sum(pi.rewards[pi.rewards >= 0])
            negative_hits = np.sum(pi.rewards[pi.rewards < 0])
            pi.onpolicy_reset()
            print('Episode %d loss %f total_reward %f positive_reward %d negative_reward %d' % (i, loss, total_reward,
                                                                                            positive_hits, negative_hits))
            losses.append(loss.item())
            total_rewards.append(total_reward)

    if pi is not None:
        return env_list, losses, total_rewards
    else:
        return env_list


def plot_rewards(controller_name, env_list):
    topN_rewards = get_rewards(env_list)
    plt.plot(topN_rewards)
    plt.title('%s Reward per Episode' % controller_name)
    plt.ylabel('Reward (cov_prop * intensity_prop)')
    plt.xlabel('Episode')


def get_rewards(env_list):
    rewards = []
    for env in env_list:
        res = evaluate_simulated_env(env)
        reward = res['coverage_proportion'] * res['intensity_proportion']
        rewards.append(reward)
    return rewards


rl_controller_name = 'REINFORCE'
rl_env_list, losses, total_rewards = run_experiment(datasets, rl_controller_name, write_mzML=False, pbar=False, early_stop=early_stop)