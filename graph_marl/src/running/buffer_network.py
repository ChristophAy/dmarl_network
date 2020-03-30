import torch as th
import numpy as np
import pdb
import dgl
from graph_marl.src.utils.helper_funs import compute_benchmark

class ReplayBuffer():

    def __init__(self, buffer_size=50, random_sample=False,
                 signal_var=1, reward_rel_benchmark=False):

        self.buffer = {}
        self.buffer_size = buffer_size
        self.network_history = {}
        self.random_sample = random_sample
        self.signal_var = signal_var
        self.reward_rel_benchmark = reward_rel_benchmark

    def add_sample(self, agent_id, sample, iteration):
        if iteration not in self.buffer:
            self.buffer[iteration] = {}
        if agent_id not in self.buffer[iteration]:
            self.buffer[iteration][agent_id] = []

        self.buffer[iteration][agent_id].append(
            sample
        )

    def add_network(self, network, iteration):
        self.network_history[iteration] = network

    def draw_sample_by_id_time(self, agent_id, time):
        if self.random_sample:
            sample_iteration = np.random.choice(
                list(self.buffer.keys())
            )
        else:
            sample_iteration = max(list(self.buffer.keys()))
        return self.buffer[sample_iteration][agent_id][time]

    def draw_sample(self):
        if self.random_sample:
            sample_iteration = np.random.choice(
                list(self.buffer.keys())
            )
        else:
            sample_iteration = max(list(self.buffer.keys()))
        return self.buffer[sample_iteration], self.network_history[sample_iteration]

    def reset_buffer(self):
        if len(self.buffer) >= self.buffer_size:
            oldest_iteration = min(list(self.buffer.keys()))
            _ = self.buffer.pop(oldest_iteration)
            _ = self.network_history.pop(oldest_iteration)

    def compute_average_reward(self, debug=False):
        last_iteration = max(list(self.buffer.keys()))
        rewards = []
        for agent in self.buffer[last_iteration]:
            reward_ts = []
            for sample in self.buffer[last_iteration][agent]:
                reward_ts.append(sample['reward'].numpy())
            rewards.append(reward_ts)
        rewards = np.array(rewards)

        # attacker_idx = self.network_history[last_iteration]['attacker_idx']
        
        # graph = self.network_history[last_iteration]['graph']
        # n_agents = dgl.unbatch(graph)[0].number_of_nodes()
        # degrees = graph.in_degrees().reshape(-1, n_agents).detach().cpu().numpy()
        # degrees_attacker = 1.0 * np.take_along_axis(degrees, attacker_idx, axis=1)
        # attacker_mean_degree = np.mean(degrees_attacker, axis=1)
        # acc_last = np.mean(rewards[:, -1, :], axis=0)

        # degree_acc_corr = np.corrcoef(attacker_mean_degree, acc_last)[0, -1]
        degree_acc_corr = 0

        frac_wrong_frac = np.array([
            [10, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.1)],
            [25, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.25)],
            [50, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.5)],
            [75, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.75)],
            [90, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.9)],
            [99, np.mean(rewards[:, -1, :].mean(axis=0) <= 0.99)],

        ])

        p_q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        quantiles = np.quantile(
            rewards[:, -1, :].mean(axis=0),
            p_q
        )
        quantiles = np.stack([p_q, quantiles]).T
        
        if debug:
            print(rewards)
        
        if not self.reward_rel_benchmark:
            mean = np.mean(rewards, axis=(0, 2))
            sd_error = np.std(rewards, axis=(0, 2)) / np.sqrt(rewards.shape[2])
            return mean, sd_error, frac_wrong_frac, quantiles, degree_acc_corr
        else:
            n_agents = dgl.unbatch(self.network_history[last_iteration]['graph'])[0].number_of_nodes()
            benchmark = compute_benchmark(n_agents, self.signal_var)
            mean = np.mean(rewards, axis=(0, 2)) / benchmark
            sd_error = np.std(rewards, axis=(0, 2)) / benchmark / np.sqrt(rewards.shape[2])

            return mean, sd_error, frac_wrong_frac, quantiles, degree_acc_corr