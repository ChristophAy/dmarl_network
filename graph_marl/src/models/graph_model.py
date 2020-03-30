import dgl.function as fn
import torch
import numpy as np
from graph_marl.src.utils.helper_funs import to_onehot

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(torch.nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class NodeEmbedding(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, device, embedding_dim=2, k_hops=2):
        super(NodeEmbedding, self).__init__()

        module_list = []
        for k in range(k_hops):
            if k == 0:
                layer = GCN(in_dim, hidden_dim, torch.nn.functional.relu)
            elif k == k_hops - 1:
                layer = GCN(hidden_dim, embedding_dim, torch.nn.functional.relu)
            else:
                layer = GCN(hidden_dim, hidden_dim, torch.nn.functional.relu)
            module_list.append(layer)
        
        self.layers = torch.nn.ModuleList(module_list)
        self.device = device

        self.linear = torch.nn.Linear(in_dim + hidden_dim * (k_hops - 1) + embedding_dim, embedding_dim)

    def forward(self, g, features=None):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h1 = g.in_degrees().view(-1, 1).float()
        h1 = h1.to(self.device)
        if features is not None:
            features = features.to(self.device)
            h1 = torch.cat([h1, features], axis=1)
        hs = [h1]
        for conv in self.layers:
            h = conv(g, hs[-1])
            hs.append(h)
        # import pdb; pdb.set_trace()
        out = self.linear(
            torch.cat(
              hs, axis=1
            )
        )
        return out

class DummyEmedding(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, embedding_dim=2, k_hops=2):
        super(DummyEmedding, self).__init__()

        self.embedding_dim = embedding_dim

    def forward(self, graph, features=None):
        dummy_embedding = torch.tensor(
            np.random.normal(size=(graph.nodes().size(0), self.embedding_dim))
        )
        dummy_embedding = dummy_embedding.float()
        return dummy_embedding

class IdentityEmedding(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, embedding_dim=2, k_hops=2):
        super(IdentityEmedding, self).__init__()

        self.embedding_dim = embedding_dim

    def forward(self, g, features=None):
        h1 = g.in_degrees().view(-1, 1).float()

        return h1


class TypeEmedding(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, label, embedding_dim=2, k_hops=2):
        super(TypeEmedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.label = label

    def forward(self, g, features=None):
        h1 = g.ndata[self.label].view(-1, 1).float()
        # h1 = to_onehot(h1)

        return h1