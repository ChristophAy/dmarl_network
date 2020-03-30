import torch as th
from torch.nn.functional import softmax
from torch.nn.functional import relu
import pdb

class GRU_model_attention(th.nn.Module):

    def __init__(self, obs_dim, n_hidden, n_layers,
                 num_actions, max_degree, obs_idx, T, device, signal_dim=1, 
                 full_act_obs=None,
                 weight_features='degree', weight_mode='static'):

        super(GRU_model_attention, self).__init__()

        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_actions = num_actions
        self.max_degree = max_degree
        self.obs_idx = obs_idx
        self.T = T
        self.weight_features = weight_features
        self.weight_mode = weight_mode
        self.device = device

        if self.weight_mode == 'dynamic':
            if self.weight_features == 'degree':
                self.weights_gru = th.nn.GRU(self.max_degree, self.n_hidden, self.n_layers)
            elif self.weight_features == 'all':
                self.weights_gru = th.nn.GRU(self.obs_dim, self.n_hidden, self.n_layers)
            else:
                raise NotImplementedError
            if full_act_obs is None:
                self.weights_linear = th.nn.Linear(self.n_hidden, self.max_degree)
            else:
                self.weights_linear = th.nn.Linear(self.n_hidden, full_act_obs)
        elif self.weight_mode == 'static':
            if self.weight_features == 'degree':
                self.weights_net = th.nn.Linear(self.max_degree, self.max_degree)
            elif self.weight_features == 'all':
                self.weights_net = th.nn.Linear(self.obs_dim, self.max_degree)
            else:
                raise NotImplementedError
        elif self.weight_mode == 'equal':
            pass
        elif self.weight_mode == 'kernel':
            # dimension: embedding_node, embedding_neighbor, action_neighbor
            self.node_embedding_dim = 2
            self.action_embedding_dim = 2
            self.kernel = th.nn.GRU(2 * self.node_embedding_dim + 1, self.n_hidden, self.n_layers)
        else:
            raise NotImplementedError
        
        
        # Input: [signal, weighted neigbor actions, time stamp]
        # self.gru_qval = th.nn.GRU(2 + self.T, self.n_hidden, self.n_layers)
        # Input: [signal, weighted neigbor actions, own_degree, time stamp]
        if self.weight_mode == 'kernel':
            self.gru_qval = th.nn.GRU(self.n_hidden + 1 + self.T, self.n_hidden, self.n_layers)
        else:
            self.gru_qval = th.nn.GRU(2 + signal_dim + self.T, self.n_hidden, self.n_layers)
        self.linear_qval = th.nn.Linear(self.n_hidden, self.num_actions)

    def forward(self, observation, init_rnn_state=None):

        observation.to(self.device)
        
        # Extract action observation embeddings and node embeddings.
        obs_tensor = th.cat(
            [observation[:, :, 0:self.obs_idx[0]],
            observation[:, :, self.obs_idx[1]:]], axis=2
        )

        if init_rnn_state is None:
            init_rnn_state = {
                'attention' : None,
                'qval' : None
            }
        else:
            if init_rnn_state['attention'] is not None:
                init_rnn_state['attention'].to(self.device)
            if init_rnn_state['qval'] is not None:
                init_rnn_state['qval'].to(self.device)

        next_rnn_state = {
                'attention' : None,
                'qval' : None
            }
        
        # Numerical values for actions
        obs_actions = observation[:, :, self.obs_idx[0]:self.obs_idx[1]]
        # Signal
        # signal = observation[:, :, self.obs_idx[2]][:, :, None]
        signal = observation[:, :, self.obs_idx[2]:self.obs_idx[3]]
        # Time embedding
        # time = observation[:, :, (self.obs_idx[2]+1):self.obs_idx[0]]
        time = observation[:, :, (self.obs_idx[3]):self.obs_idx[0]]
        # Degree of neighbors -> this is only relevant for case with IdentityEmbedder
        neighbor_degrees = observation[:, :, self.obs_idx[1]+1:]

        if self.weight_features == 'degree':
            in_ = neighbor_degrees
        elif self.weight_features == 'all':
            in_ = obs_tensor

        if self.weight_mode == 'dynamic':
            gru_output, next_rnn_state_a = self.weights_gru(in_, init_rnn_state['attention'])
            weight_logits = self.weights_linear(gru_output)
            next_rnn_state['attention'] = next_rnn_state_a
        elif self.weight_mode == 'static':
            weight_logits = self.weights_net(in_)
            # weight_logits = in_ * 0.1
        elif self.weight_mode == 'equal':
            weight_logits = th.zeros(size=obs_actions.shape)
            weight_logits[obs_actions != 0] = 1
        elif self.weight_mode == 'kernel':
            batch_size = in_.shape[1]
            gru_output = []
            next_rnn_state_a = []
            neighbor_embedding = observation[:, :, self.obs_idx[1]+2:]
            self_embedding = observation[:, :, self.obs_idx[1]:self.obs_idx[1]+2]
            for b in range(batch_size):
                x = neighbor_embedding[:, b, :][:, None, :]
                x = x[x != 0]
                x = x.reshape(in_.shape[0], 1, -1)
                x = x.reshape(in_.shape[0], -1, self.node_embedding_dim)
                y = self_embedding[:, b, :][:, None, :]
                z = obs_actions[:, b, :][:, None, :]
                z = z[z != 0]
                z = z.reshape(in_.shape[0], -1, 1)
                z = th.cat([x, z], axis=2)
                ys = []
                for _ in range(z.shape[1]):
                    ys.append(y)
                y = th.cat(ys, axis=1)
                z = th.cat([y, z], axis=2)
                if init_rnn_state['attention'] is None:
                    g, nr = self.kernel(z, None)
                else:
                    g, nr = self.kernel(z, init_rnn_state['attention'][b])
                next_rnn_state_a.append(nr)
                gru_output.append(g.mean(axis=1))
            next_rnn_state['attention'] = next_rnn_state_a
            
            gru_output = th.stack(gru_output, axis=0)
            gru_output = gru_output.permute(1, 0, 2)

            weight_logits = th.zeros(size=obs_actions.shape)
            weight_logits[obs_actions != 0] = 1

        own_degree = (obs_actions != 0).sum(axis=2, keepdim=True).float()

        weight_logits[obs_actions == 0] = float('-inf')
        weights = softmax(weight_logits, dim=2)

        weighted_actions = (obs_actions * weights).sum(axis=2, keepdim=True)
        # in_qval_ = th.cat([weighted_actions, signal, time], axis=2)
        in_qval_ = th.cat([weighted_actions, signal, time, own_degree], axis=2)
        if self.weight_mode == 'kernel':
            in_qval_ = th.cat([gru_output, signal, time], axis=2)
        else:
            in_qval_ = th.cat([weighted_actions, signal, time, own_degree], axis=2)
        
        gru_output_qval, next_rnn_state_qval = self.gru_qval(in_qval_, init_rnn_state['qval'])
        qvals = self.linear_qval(gru_output_qval)

        next_rnn_state['qval'] = next_rnn_state_qval

        return qvals, next_rnn_state