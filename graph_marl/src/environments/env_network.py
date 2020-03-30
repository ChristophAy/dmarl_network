import pdb
import random
import numpy as np
import torch
import os
import dgl
import networkx as nx
import pickle
from collections import OrderedDict

from graph_marl.src.utils.helper_funs import encode_and_concat_observations
from graph_marl.src.utils.helper_funs import to_onehot
from graph_marl.src.utils.helper_funs import make_neighborhood_embedding, compute_laplacian_eigenvectors
from graph_marl.src.utils.network_generators import make_fb_nets, make_core_periphery, make_barabasi_albert_network, make_degree_effect_graph, make_two_star

class env_multi():
    
    def __init__(self, node_embedder, agent_type_dict, args):

        self.args = args
        
        self.var = args.var
        self.n_states = args.n_states
        self.n_batches = args.n_batches
        self.batch_networks = not args.no_batch_networks
        self.min_agents = args.min_agents
        self.max_agents = args.max_agents
        self.max_degree = args.max_degree
        self.network_type = args.network_type

        self.num_actions = args.num_actions
        self.node_embedder = node_embedder
        self.agent_type_dict = agent_type_dict
        self.do_neighborhood_embedding = args.do_neighborhood_embedding
        self.k_laplacian_eigenvectors = args.input_dim_node_embedding - 1
        self.use_attention_net = args.use_attention_net

        self.attack = args.attack
        self.bias_per_agent = args.bias_per_agent
        self.n_attack = args.n_attack
        self.trained_attacker = args.trained_attacker

        self.type_label = args.type_label
        self.signal_var_types = args.signal_var_types
        self.low_precision_var_factor = args.low_precision_var_factor
        self.n_h_type = args.n_h_type
        self.n_core = args.n_core
        self.use_kernel_net = args.use_kernel_net

        self.attack_full_action_obs = args.attack_full_action_obs
        self.attack_full_signal_obs = args.attack_full_signal_obs
        self.fixed_node_encoding = args.fixed_node_encoding

        if self.signal_var_types and self.network_type != 'core_periphery':
            print('Signal var types only supported for CP nets. Reverting...')
            self.signal_var_types = False

        if self.network_type == 'facebook':
            network_path = os.path.abspath(
                os.path.join('social_network_files', 'facebook_networks'))
            network_file = '107.edges'
            edges = np.loadtxt(
                os.path.join(network_path, network_file), dtype=np.int32)
            self.facebook_graph = nx.from_edgelist(edges)
            A = nx.adj_matrix(self.facebook_graph)
            A = A.todense()
            A = np.array(A, dtype = np.float64)
            # let's evaluate the degree matrix D
            Dinv = np.diag(1 / np.sum(A, axis=1))
            # ...and the transition matrix T
            self.facebook_Tm = np.dot(Dinv,A)
        elif self.network_type == 'stored_sample':
            with open(os.path.join(self.args.network_path, self.args.network_file), 'rb') as f:
                self.stored_sample = pickle.load(f)
        elif self.network_type == 'from_file':
            self.stored_sample = self.convert_neighborhoods_to_dgl_graph(
                os.path.join(self.args.network_path, self.args.network_file)
            )
                    
        self.reset()

    def make_network_instance(self):

        if self.network_type == 'from_file':
            graph = self.stored_sample
        elif self.network_type == 'stored_sample':
            graph = self.stored_sample
        elif self.network_type == 'core_periphery':
            # n_core = int(0.3 * self.n_agents)
            graph, _ = make_core_periphery(
                n=self.n_agents, n_core=self.n_core, n_h=self.n_h_type
            )
        elif self.network_type == 'degree_effect':
            graph = make_degree_effect_graph(n=self.n_agents)
        elif self.network_type == 'two_star':
            graph = make_two_star(n=self.n_agents)
        elif self.network_type == 'barabasi_albert':
            graph = make_barabasi_albert_network(n=self.n_agents)
        elif self.network_type == 'facebook':
            graph = make_fb_nets(self.facebook_graph, self.facebook_Tm, n=self.n_agents, 
                                 max_degree=self.max_degree)
        else:
            print('Invalid network type')
            raise NotImplementedError
        
        return graph

    def reset(self):
        
        if self.network_type == 'stored_sample':
            self.n_agents = self.stored_sample.number_of_nodes()
        else:
            self.n_agents = np.random.randint(self.min_agents, self.max_agents + 1)
        
        self.n_agents_by_type = OrderedDict()
        for a_type in self.agent_type_dict:
            if a_type == 'citizen':
                if self.trained_attacker:
                    self.n_agents_by_type[a_type] = self.n_agents - self.n_attack
                else:
                    self.n_agents_by_type[a_type] = self.n_agents
            elif a_type == 'attacker':
                self.n_agents_by_type[a_type] = self.n_attack
            else:
                print('Currently only supporting citizen or attacker types.')
                raise NotImplementedError

        self.T = self.args.T

        if self.batch_networks:
            graphs = []
            neighborhoods = []
            laplacian_eigenvectors = []
            degrees =  []
            var_types = []
            for _ in range(self.n_batches):
                graph = self.make_network_instance()
                neighborhood = self.make_neighboords(graph)
                _, eigenVectors, _ = compute_laplacian_eigenvectors(graph)
                degree = np.array(list(dict(graph.to_networkx().in_degree()).values()))
                if self.signal_var_types:
                    typ = graph.ndata[self.type_label].cpu().numpy()
                    var_type = np.ones((1, graph.number_of_nodes())) * self.var * self.low_precision_var_factor
                    var_type[0, typ == 1] = self.var
                    var_types.append(var_type)

                degrees.append(degree)
                graphs.append(graph)
                neighborhoods.append(neighborhood)
                if self.k_laplacian_eigenvectors > 0:
                    laplacian_eigenvectors.append(
                        torch.tensor(eigenVectors[:, 0:self.k_laplacian_eigenvectors]).float()
                    )
            self.graph = dgl.batch(graphs)
            self.neighborhoods = neighborhoods
            self.degree_matrix = np.stack(degrees)
            if self.signal_var_types:
                self.var_types = np.concatenate(var_types, axis=0)
            if self.k_laplacian_eigenvectors > 0:
                self.net_features = torch.cat(
                    laplacian_eigenvectors, axis=0
                )
            else:
                self.net_features = None
        else:
            graph = self.make_network_instance()
            neighborhood = self.make_neighboords(graph)
            _, eigenVectors, _ = compute_laplacian_eigenvectors(graph)
            graphs = []
            neighborhoods = []
            laplacian_eigenvectors = []
            for _ in range(self.n_batches):
                graphs.append(graph)
                neighborhoods.append(neighborhood)
                if self.k_laplacian_eigenvectors > 0:
                    laplacian_eigenvectors.append(
                        torch.tensor(eigenVectors[:, 0:self.k_laplacian_eigenvectors]).float()
                    )
            self.graph = dgl.batch(graphs)
            self.neighborhoods = neighborhoods
            if self.k_laplacian_eigenvectors > 0:
                self.net_features = torch.cat(
                    laplacian_eigenvectors, axis=0
                )
            else:
                self.net_features = None

        self.agents_by_type = OrderedDict()
        for a_type in self.n_agents_by_type:
            self.agents_by_type[a_type] = OrderedDict()
            for i in range(self.n_agents_by_type[a_type]):
                self.agents_by_type[a_type][i] = self.agent_type_dict[a_type]
        
        self.time_step = 0
        self.world = np.random.randint(
            0, self.n_states, size=(self.n_batches, 1)
            )

        if self.signal_var_types:
            errs = np.random.normal(
                size=(self.n_batches, self.n_agents)
                ) * np.sqrt(self.var_types)
        else:
            errs = np.random.normal(
                size=(self.n_batches, self.n_agents)
                ) * np.sqrt(self.var)

        self.signals = errs + self.world

        if self.attack:
            self.attack_idx = np.zeros((self.n_batches, self.n_attack), dtype=np.int)
            for i in range(self.n_batches):
                idx = np.random.choice(
                    np.arange(0, self.n_agents), size=self.n_attack, replace=False)
                self.attack_idx[i, :] = idx
                for j in idx:
                    if self.trained_attacker:
                        # trained attacker knows the true state of the world.
                        self.signals[i, j] = self.world[i, 0]
                    else:
                        self.signals[i, j] += self.bias_per_agent * (1 - 2 * self.world[i, 0])

        self.idx_type_map = OrderedDict()
        for a_type in self.agent_type_dict:
            self.idx_type_map[a_type] = OrderedDict()
            if a_type == 'citizen':
                for b in range(self.n_batches):
                    if self.trained_attacker:
                        idx = [i for i in range(self.n_agents) if i not in self.attack_idx[b, :]]
                        self.idx_type_map[a_type][b] = np.array(idx)
                    else:
                        self.idx_type_map[a_type][b] = np.arange(self.n_agents)
            elif a_type == 'attacker':
                for b in range(self.n_batches):
                    self.idx_type_map[a_type][b] = self.attack_idx[b, :]
            else:
                print('Only citizen and attacker types currently supporter')
                raise NotImplementedError
        
        self.inverse_idx_type_map = OrderedDict()
        for b in range(self.n_batches):
            cnt_type = {
                'attacker' : 0,
                'citizen' : 0
            }
            self.inverse_idx_type_map[b] = OrderedDict()
            for i in range(self.n_agents):
                if self.attack:
                    if i in self.attack_idx[b, :] and self.trained_attacker:
                        self.inverse_idx_type_map[b][i] = ('attacker', cnt_type['attacker'])
                        cnt_type['attacker'] += 1
                    else:
                        self.inverse_idx_type_map[b][i] = ('citizen', cnt_type['citizen'])
                        cnt_type['citizen'] += 1
                else:
                    self.inverse_idx_type_map[b][i] = ('citizen', cnt_type['citizen'])
                    cnt_type['citizen'] += 1


        self.node_embeddings = self.node_embedder(
            self.graph, features=self.net_features
        ).reshape((self.n_batches, self.n_agents, -1))
        self.neighborhood_embedding_by_agents = make_neighborhood_embedding(
            self.node_embeddings, self.neighborhoods,self.max_degree, add_neighbors=self.do_neighborhood_embedding
        )

        samples = OrderedDict()
        for a_type in self.agent_type_dict:
            samples[a_type] = OrderedDict()
            for agent in range(self.n_agents_by_type[a_type]):
                samples[a_type][agent] =  {
                    'action' : torch.tensor(
                        np.zeros((self.n_batches, ), dtype=int)
                    )
                }

        if self.use_kernel_net:
            agent_observations, neighborhood_embedding_by_agents, _, _ = self.make_agent_obs_kernel(samples)
        else:
            agent_observations, neighborhood_embedding_by_agents, _ = self.make_agent_obs(samples)

        return agent_observations, neighborhood_embedding_by_agents, self.node_embeddings

    def make_agent_obs(self, samples):

        agent_observations = OrderedDict()
        rewards = OrderedDict()
        for a_type in self.agent_type_dict:
            agent_observations[a_type] = OrderedDict()
            rewards[a_type] = OrderedDict()
            for i in range(self.n_agents_by_type[a_type]):
                if a_type == 'attacker' and self.attack_full_action_obs:
                    observed_actions = np.zeros((self.n_batches, self.n_agents))
                    observed_actions_num = np.zeros((self.n_batches, self.n_agents))
                else:
                    observed_actions = np.zeros((self.n_batches, self.max_degree))
                    observed_actions_num = np.zeros((self.n_batches, self.max_degree))
                signal = np.zeros((self.n_batches, 1))
                reward = np.zeros((self.n_batches, ), dtype=np.float32)
                for b in range(self.n_batches):
                    idx = self.idx_type_map[a_type][b][i]
                    act = samples[a_type][i]['action'].cpu().numpy()[b]
                    observed_actions[b, 0] = act + 1
                    observed_actions_num[b, 0] = 2 * act - 1

                    if a_type == 'attacker' and self.attack_full_action_obs:
                        for k in range(self.n_agents):
                            neigh_type, idx_type = self.inverse_idx_type_map[b][k]
                            act = samples[neigh_type][idx_type]['action'].cpu().numpy()[b]
                            observed_actions[b, k] = act + 1
                            observed_actions_num[b, k] = 2 * act - 1
                    else:
                        for k, neigh in enumerate(self.neighborhoods[b][idx]):
                            neigh_type, idx_type = self.inverse_idx_type_map[b][neigh]
                            act = samples[neigh_type][idx_type]['action'].cpu().numpy()[b]
                            observed_actions[b, k] = act + 1
                            observed_actions_num[b, k] = 2 * act - 1

                    signal[b, 0] = self.signals[b, idx]
                    if a_type == 'citizen':
                        act = samples[a_type][i]['action'].cpu().numpy()[b]
                        if act == self.world[b, 0]:
                            reward[b] = 1
                    elif a_type == 'attacker':
                        r_citizen = 0
                        for j in range(self.n_agents):
                            if j in self.idx_type_map['citizen'][b]:
                                _, idx_j = self.inverse_idx_type_map[b][j]
                                act = samples['citizen'][idx_j]['action'].cpu().numpy()[b]
                                if act == self.world[b, 0]:
                                    r_citizen += 1
                        r_citizen = r_citizen / self.n_agents_by_type['citizen']
                        reward[b] = - r_citizen
                    else:
                        print('Currently only supporting citizen or attacker types.')
                        raise NotImplementedError

                rewards[a_type][i] = torch.tensor(reward)

                obs = OrderedDict()
                obs_encoding = OrderedDict()

                obs['actions'] = observed_actions

                if self.fixed_node_encoding:
                    obs['fixed_node_encoding'] = np.ones((self.n_batches, 1), dtype=np.float64) * i
                    obs_encoding['fixed_node_encoding'] = ('one_hot', self.n_agents_by_type[a_type])

                obs['signal'] = signal
                if a_type == 'attacker' and self.attack_full_signal_obs:
                    all_signals = self.signals.reshape(self.n_batches, -1)
                    obs['signal'] = all_signals

                obs['time'] = np.ones(shape=self.signals[:, i, None].shape) * self.time_step

                obs_encoding['actions'] = ('one_hot', self.num_actions + 1)
                obs_encoding['signal'] = ('float', None)
                obs_encoding['time'] = ('one_hot', self.T)
                
                if self.use_attention_net:
                    obs['actions_float'] = observed_actions_num
                    obs_encoding['actions_float'] = ('float', None)

                agent_observations[a_type][i] = encode_and_concat_observations(obs, obs_encoding)

        return agent_observations, self.neighborhood_embedding_by_agents, rewards
                
    def step(self, samples):
        
        if self.use_kernel_net:
            agent_observations, neighborhood_embedding_by_agents, rewards, _ = self.make_agent_obs_kernel(samples)
        else:
            agent_observations, neighborhood_embedding_by_agents, rewards = self.make_agent_obs(samples)
        self.time_step += 1
        
        done = torch.ones((self.n_batches), dtype=torch.int32) if self.time_step == self.T else torch.zeros((self.n_batches), dtype=torch.int32)
        
        return agent_observations, neighborhood_embedding_by_agents, rewards, done

    def load_network(self, filename):

        f = open(filename, 'r')
        neighborhoods = dict() 
        cnt = 0
        for l in f:
            tmp = [int(i) for i in l.split(',')]
            neighborhoods[cnt] = tmp[1:]
            cnt += 1

        return neighborhoods

    def convert_neighborhoods_to_dgl_graph(self, filename):

        G = nx.DiGraph()
        with open(filename, "r") as text_file:
            for line in text_file:
                nodes = line.strip('\n').split(',')
                # add self edge
                G.add_edge(int(nodes[0]), int(nodes[0]))
                # add all other edges
                if len(nodes) > 1:
                    for n in nodes[1:]:
                        G.add_edge(int(nodes[0]), int(n))
                else:
                    pass
                    # G.add_node(nodes)
        graph = dgl.DGLGraph(G)
        return graph

    def make_neighboords(self, graph):

        G = graph.to_networkx()

        network = OrderedDict()

        for n1 in G.nodes():
            network[n1] = []
            for n2 in G[n1]:
                network[n1].append(n2)

        neighborhoods = []
        for j in range(self.n_agents):
            neighborhood = network[j]
            neighborhoods.append(
                [neighborhood[i] for i in range(len(neighborhood))]
                )

        return neighborhoods

    def make_agent_obs_kernel(self, samples):

        agent_observations = OrderedDict()
        rewards = OrderedDict()
        for a_type in self.agent_type_dict:
            agent_observations[a_type] = OrderedDict()
            rewards[a_type] = OrderedDict()
            for i in range(self.n_agents_by_type[a_type]):
                signal = np.zeros((self.n_batches, 1))
                reward = np.zeros((self.n_batches, ), dtype=np.float32)
                M = []
                S = []
                cnt = 0
                for b in range(self.n_batches):
                    # in the first entry observe own past action
                    idx = self.idx_type_map[a_type][b][i]

                    for k, neigh in enumerate(self.neighborhoods[b][idx]):
                        neigh_type, idx_type = self.inverse_idx_type_map[b][neigh]

                        self_act = torch.tensor(samples[a_type][i]['action'].cpu().numpy()[b])
                        neig_act = torch.tensor(samples[neigh_type][idx_type]['action'].cpu().numpy()[b])
                        self_act = to_onehot(self_act, self.num_actions + 1).float()
                        neig_act = to_onehot(neig_act, self.num_actions + 1).float()

                        # self_emb = self.node_embeddings[b, i, :]
                        # neig_emb = self.node_embeddings[b, neigh, :]

                        self_emb = torch.tensor([i]).float()
                        neig_emb = torch.tensor([neigh]).float()
                        batch = torch.tensor([b]).float()

                        M.append(
                            torch.cat([self_act, neig_act, batch, self_emb, neig_emb], axis=0)[None, :]
                        )
 
                    selector = np.zeros((cnt + len(self.neighborhoods[b][idx]), 1))
                    selector[cnt:, 0] = 1
                    S.append(
                        torch.tensor(selector).float()
                    )
                    cnt += len(self.neighborhoods[b][idx])

                    signal[b, 0] = self.signals[b, idx]
                    if a_type == 'citizen':
                        act = samples[a_type][i]['action'].cpu().numpy()[b]
                        if act == self.world[b, 0]:
                            reward[b] = 1
                    elif a_type == 'attacker':
                        r_citizen = 0
                        for j in range(self.n_agents):
                            if j in self.idx_type_map['citizen'][b]:
                                _, idx_j = self.inverse_idx_type_map[b][j]
                                act = samples['citizen'][idx_j]['action'].cpu().numpy()[b]
                                if act == self.world[b, 0]:
                                    r_citizen += 1
                        r_citizen = r_citizen / self.n_agents_by_type['citizen']
                        reward[b] = - r_citizen
                    else:
                        print('Currently only supporting citizen or attacker types.')
                        raise NotImplementedError

                rewards[a_type][i] = torch.tensor(reward)
                
                for b in range(len(S)):
                    S[b] = torch.nn.functional.pad(
                        S[b].T, (0, S[-1].shape[0] - S[b].shape[0])
                    ).T
                S = torch.cat(S, axis=1)
                M = torch.cat(M, axis=0)
                time = torch.tensor(np.ones(shape=self.signals[:, i, None].shape) * self.time_step)
                time = to_onehot(time, self.T).float()
                signal = torch.tensor(signal).float()
                agent_observations[a_type][i] = [{'M' : M, 'S' : S, 'signal' : signal, 'time' : time}]

        return agent_observations, self.neighborhood_embedding_by_agents, rewards, self.node_embeddings
