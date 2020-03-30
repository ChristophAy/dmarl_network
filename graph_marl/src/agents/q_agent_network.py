import torch as th
import pdb
import numpy as np
from copy import deepcopy
from graph_marl.src.utils.helper_funs import select_at_indexes
from graph_marl.src.utils.helper_funs import cat_obs_node_embedding
from graph_marl.src.models.gru_model import GRU_model
from graph_marl.src.models.attention_gru_model import GRU_model_attention
from graph_marl.src.models.graph_model import NodeEmbedding

class RQ_agent(object):

    def __init__(self, n_agents, obs_dim, n_hidden, n_layers, num_actions,
                 max_degree, T, exploration_rate, device, obs_idx=None, use_attention_net=False,
                 use_kernel_net=False, node_embedding_dim=2,
                 attention_weight_mode=None,
                 attention_weight_features=None,
                 boltzmann=False, temperature=0, 
                 signal_dim=1,
                 full_act_obs=None):
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.exploration_rate = exploration_rate
        self.use_attention_net = use_attention_net
        self.use_kernel_net = use_kernel_net
        self.node_embedding_dim = node_embedding_dim
        self.boltzmann = boltzmann
        self.temperature = temperature
        self.T = T
        self.device=device

        self.rnn_state = {}
        for i in range(self.n_agents):
            self.rnn_state[i] = None
            # TODO: does it make sense for this to be held here?

        if self.use_attention_net:
            self.q_train = GRU_model_attention(self.obs_dim, n_hidden, n_layers, num_actions, max_degree,
                                               obs_idx, T, device, weight_features=attention_weight_features,
                                               signal_dim=signal_dim,
                                               full_act_obs=full_act_obs,
                                               weight_mode=attention_weight_mode)
            self.q_target = GRU_model_attention(self.obs_dim, n_hidden, n_layers, num_actions, max_degree,
                                                obs_idx, T, device, weight_features=attention_weight_features,
                                                signal_dim=signal_dim,
                                                full_act_obs=full_act_obs,
                                                weight_mode=attention_weight_mode)
        else:
            self.q_train = GRU_model(self.obs_dim, n_hidden, n_layers, num_actions)
            self.q_target = GRU_model(self.obs_dim, n_hidden, n_layers, num_actions)

        self.q_train.to(device)
        self.q_target.to(device)

    @th.no_grad()
    def take_action(self, agent_idx, observation, neighborhood_embedding_by_agents, node_embeddings, timestep, testing=False):
        
        # Observation dimensions are [T, B, F] = [Timesteps (just 1), Batchsize, Feature_dimension]
        if self.use_kernel_net:
            batch_size = observation[0]['signal'].shape[0]
        else:
            batch_size = observation.size(1)

        eps = self.exploration_rate if not testing else 0
        prev_rnn_state = deepcopy(self.rnn_state[agent_idx])

        observation_node_embedding = cat_obs_node_embedding(
            observation.to(self.device), neighborhood_embedding_by_agents, batch_size, use_kernel_net=self.use_kernel_net,
            node_embeddings=node_embeddings
        )

        qvals, self.rnn_state[agent_idx] = self.q_train.forward(observation_node_embedding, init_rnn_state=self.rnn_state[agent_idx])

        # if np.random.uniform(0, 1) < eps and not testing:
        #     action = th.tensor(np.random.randint(0, self.num_actions, size=batch_size))
        # else:
        #     action = th.argmax(qvals, dim=2)[0, :]
        
        if not testing:
            if self.boltzmann:
                probs = th.softmax(self.temperature * qvals, dim=2)
                dist = th.distributions.categorical.Categorical(probs=probs[0, :, :])
                action = dist.sample()
            else:
                mask = th.zeros(batch_size, dtype=th.long)
                mask[np.random.uniform(0, 1, size=batch_size) < eps] = 1
                random_action = th.tensor(np.random.randint(0, self.num_actions, size=batch_size))
                max_action = th.argmax(qvals, dim=2)[0, :]
                action = mask * random_action + (1 - mask) * max_action
        else:
            if self.boltzmann:
                probs = th.softmax(self.temperature * qvals, dim=2)
                dist = th.distributions.categorical.Categorical(probs=probs[0, :, :])
                action = dist.sample()
            else:
                action = th.argmax(qvals, dim=2)[0, :]

        return action, prev_rnn_state, qvals

    def update_target(self):
        self.q_target.load_state_dict(
            self.q_train.state_dict())

    def reset_rnn_state(self):
        self.rnn_state = {}
        for i in range(self.n_agents):
            self.rnn_state[i] = None

    def set_train(self):
        self.q_train.train()
        self.q_target.train()

    def set_eval(self):
        self.q_train.eval()
        self.q_target.eval()

    def get_models(self):

        models = {}
        models['q_train'] = self.q_train
        models['q_target'] = self.q_target

        return models





