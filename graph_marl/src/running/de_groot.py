import networkx as nx
from copy import deepcopy
import numpy as np
import pdb
from scipy.stats import norm
import dgl

class DeGrootBenchmark():

    def __init__(self, environment, T, normalize):

        self.env = environment
        self.T = T
        self.normalize = normalize

    def run_episode(self):

        _, _, _ = self.env.reset()
        graph_batch = dgl.unbatch(self.env.graph)
        last_accuracy = []
        for graph in graph_batch:
            A = nx.to_numpy_array(
                graph.to_networkx()
            )
            signals = np.transpose(self.env.signals)
            action = np.zeros(signals.shape)
            action[signals > 0.5] = 1

            accuracy = [(action.T == self.env.world).mean()]

            for _ in range(self.T-1):
                last_action = deepcopy(action)
                # pdb.set_trace()
                neighbor_average = A.dot(last_action) / A.sum(axis=1)[:, None]
                action = np.zeros(last_action.shape)
                action[neighbor_average > 0.5] = 1
                accuracy.append(
                    (action.T == self.env.world).mean()
                )

            benchmark = norm.cdf(0.5, loc=0, scale=np.sqrt(self.env.var / graph.number_of_nodes()) )
            if self.normalize:
                last_accuracy.append(
                    accuracy / benchmark
                )
            else:
                last_accuracy.append(accuracy)
        
        last_accuracy = np.array(last_accuracy)

        return np.mean(last_accuracy, axis=0), np.std(last_accuracy, axis=0) / np.sqrt(last_accuracy.shape[0])

