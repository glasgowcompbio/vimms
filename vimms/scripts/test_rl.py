import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

from random import randrange
import random as rand
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim

from vimms.Common import *
from vimms.Chemicals import ChemicalMixtureCreator, UniformRTAndIntensitySampler
from vimms.ChemicalSamplers import UniformMZFormulaSampler
from vimms.Evaluation import evaluate_simulated_env
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController, AgentBasedController
from vimms.Agent import TopNDEWAgent, ReinforceAgent, Pi, train, RandomAgent
from vimms.Environment import Environment
from vimms.Noise import UniformSpikeNoise

base_dir = os.path.abspath(
    'C:\\Users\\joewa\\University of Glasgow\\Vinny Davies - CLDS Metabolomics Project\\Trained Models')
hmdb = load_obj(Path(base_dir, 'hmdb_compounds.p'))

set_log_level_warning()

no_episodes = 10
n_chemicals = (100, 100)
mz_range = (100, 600)
rt_range = [(0, 300)]
intensity_range = (1E5, 1E10)

min_mz = mz_range[0]
max_mz = mz_range[1]
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]
min_log_intensity = np.log(intensity_range[0])
max_log_intensity = np.log(intensity_range[1])

datasets = []
for i in range(no_episodes):
    if i % 10 == 0: print(i)

    # df = DatabaseFormulaSampler(hmdb, min_mz=min_mz, max_mz=max_mz)
    df = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
    ris = UniformRTAndIntensitySampler(min_rt=rt_range[0][0], max_rt=rt_range[0][1],
                                       min_log_intensity=min_log_intensity, max_log_intensity=max_log_intensity)

    cm = ChemicalMixtureCreator(df, rt_and_intensity_sampler=ris)
    n_chems = n_chemicals[0]
    chems = cm.sample(n_chems, 2, include_adducts_isotopes=False)
    datasets.append(chems)

isolation_window = 0.7
N = 10
rt_tol = 5
mz_tol = 10
min_ms1_intensity = 5000
ionisation_mode = POSITIVE

in_dim = int(max_mz - min_mz)
out_dim = 10
hidden_dim = 32
gamma = 0.99
lr = 0.001

noise_density = 0.3
noise_max_val = 1e4

pbar = False
write_mzML = False


def run_experiment(datasets, controller_name, write_mzML=False, pbar=False):
    # initial setup
    pi = None
    if controller_name == 'REINFORCE':
        pi = Pi(in_dim, out_dim, hidden_dim)
        optimizer = optim.AdamW(pi.parameters(), lr=lr)

    # run episodes
    env_list = []
    losses = []
    total_rewards = []
    total_rewards_per_chems = []
    for i in range(len(datasets)):
        if i % 20 == 0:
            if pi is None: print(i)
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
            agent = TopNDEWAgent(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)
            controller = AgentBasedController(agent)

        elif controller_name == 'REINFORCE':
            agent = ReinforceAgent(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity, pi, min_mz,
                                   max_mz)
            controller = AgentBasedController(agent)

        elif controller_name == 'Random':
            num_states = out_dim
            agent = RandomAgent(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity, out_dim)
            controller = AgentBasedController(agent)

            # run the simulation
        out_dir = 'results' if write_mzML is True else None
        out_file = out_file = 'test_%s_%d.mzML' % (controller_name, i) if write_mzML is True else None

        spike_noise = UniformSpikeNoise(noise_density, noise_max_val, min_mz=min_mz, max_mz=max_mz)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, datasets[i], None, spike_noise=spike_noise)
        env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=pbar, out_dir=out_dir, out_file=out_file)
        env.run()
        env_list.append(env)

        # train REINFORCE model
        if pi is not None:
            loss = train(pi, optimizer, gamma)
            total_reward = sum(pi.rewards)
            num_chems = len(datasets[i])
            reward_per_chems = total_reward / num_chems
            print('Episode %d\tloss %8.2f\tnum_chems %4d\ttotal_reward %8.2f\ttotal_reward/num_chems %8.2f' % (
                i, loss, num_chems, total_reward, reward_per_chems))

            pi.onpolicy_reset()
            losses.append(loss.item())
            total_rewards.append(total_reward)
            total_rewards_per_chems.append(reward_per_chems)

    if pi is not None:
        return env_list, losses, total_rewards, total_rewards_per_chems, pi
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


np.random.seed(0)
rand.seed(0)

rl_controller_name = 'REINFORCE'
rl_env_list, losses, total_rewards, total_rewards_per_chems, pi = run_experiment(datasets, rl_controller_name,
                                                                                 write_mzML=False, pbar=False)
