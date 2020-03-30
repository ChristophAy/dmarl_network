import torch as th

class GRU_model(th.nn.Module):

    def __init__(self, obs_dim, n_hidden, n_layers, num_actions):

        super(GRU_model, self).__init__()

        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_actions = num_actions

        self.gru = th.nn.GRU(self.obs_dim, self.n_hidden, self.n_layers)
        self.linear = th.nn.Linear(self.n_hidden, self.num_actions)

    def forward(self, observation, init_rnn_state=None):
        
        gru_output, next_rnn_state = self.gru(observation, init_rnn_state)
        qvals = self.linear(gru_output)
        
        return qvals, next_rnn_state