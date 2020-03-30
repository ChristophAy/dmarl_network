import torch
import numpy as np
import pdb
import string
import random
import networkx as nx
import dgl
from scipy.stats import norm

def to_onehot(indexes, num, dtype=None):
    """Dimension of size num added to the end of indexes.shape."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)

    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Selected dimension of onehot is removed by argmax."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes

def select_at_indexes(indexes, tensor):
    """Leading dimensions of tensor must match dimensions of indexes."""
    dim = len(indexes.shape)

    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])

def encode_and_concat_observations(observation, encoding):
    
    obs_tensors = []
    for obs in observation:
        if encoding[obs][0] == 'one_hot':
            num = int(encoding[obs][1])
            obs_enc = to_onehot(torch.tensor(observation[obs]), num)
            B = obs_enc.size(0)
            obs_enc = obs_enc.view(B, -1)
        else:
            obs_enc = torch.tensor(observation[obs])
        obs_tensors.append(obs_enc)
    
    obs_tensor = torch.cat(obs_tensors, dim=1).unsqueeze(0).float()
    
    return obs_tensor

# def encode_and_concat_observations(observation, encoding):
    
#     obs_tensors = []
#     for obs in observation:
#         if encoding[obs][0] == 'one_hot':
#             num = int(encoding[obs][1])
#             obs_enc = to_onehot(torch.tensor(observation[obs]), num)
#             B = obs_enc.size(0)
#             obs_enc = obs_enc.view(B, -1)
#         else:
#             obs_enc = torch.tensor(observation[obs])
#         obs_tensors.append(obs_enc)
    
#     obs_tensor = torch.cat(obs_tensors, dim=1).unsqueeze(0).float()
#     obs_out = {
#         'actions' : observation['actions'],
#         'obs_tensor' : obs_tensor
#     }
    
#     return obs_out

def cat_obs_node_embedding(obs, node_embedding, n_batches, n_time_steps=1, use_kernel_net=False, node_embeddings=None):

    if not use_kernel_net:
        embedding_batched = []
        for t in range(n_time_steps):
            embedding_batched.append(node_embedding)
        embedding_batched = torch.stack(embedding_batched, axis=0)
        
        return torch.cat([obs, embedding_batched], axis=2)
    else:
        return {'obs' : obs, 'node_embeddings' : node_embeddings}
    # obs_out = {
    #     'actions' : obs['actions'],
    #     'obs_tensor' : torch.cat([obs['obs_tensor'], embedding_batched], axis=2)
    # }
    # return obs_out

def make_neighborhood_embedding(node_embeddings, neighborhoods, max_neigh_size, add_neighbors=True):

    if add_neighbors:

        neighborhood_embedding_by_agents = {}
        n_agents = node_embeddings.size(1)
        n_batches = node_embeddings.size(0)

        for i in range(n_agents):
            neb = []
            for b in range(n_batches):
                neighborhood_embedding = [node_embeddings[b, i, :]]
                for neigh in neighborhoods[b][i]:
                    neighborhood_embedding.append(node_embeddings[b, neigh, :])
                neighborhood_embedding = torch.cat(neighborhood_embedding, axis=0)
                neighborhood_embedding = torch.nn.functional.pad(
                    neighborhood_embedding, (0, node_embeddings[b, i].size(0) * (max_neigh_size + 1) - neighborhood_embedding.size(0))
                )
                neb.append(neighborhood_embedding)

            neighborhood_embedding_by_agents[i] = torch.stack(neb, axis=0)
    
    else:

        neighborhood_embedding_by_agents = node_embeddings.permute(1, 0, 2)

    return neighborhood_embedding_by_agents


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def compute_benchmark(n, variance):

    return norm.cdf(0.5, loc=0, scale=np.sqrt(variance / n) )

def compute_laplacian_eigenvectors(G):

    A = G.adjacency_matrix().to_dense().numpy()
    Dinv = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    L = np.eye(A.shape[0]) - Dinv.dot(A.dot(Dinv))
    
    eigenValues, eigenVectors = np.linalg.eig(L)
    eigenValues = np.real(eigenValues)
    eigenVectors = np.real(eigenVectors)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    if np.any(np.iscomplex(eigenVectors)):
        import pdb; pdb.set_trace()

    return eigenValues, eigenVectors, L