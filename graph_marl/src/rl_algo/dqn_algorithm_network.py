import torch
import numpy as np
from graph_marl.src.utils.helper_funs import select_at_indexes
from graph_marl.src.utils.helper_funs import cat_obs_node_embedding
from graph_marl.src.utils.helper_funs import make_neighborhood_embedding

import pdb

class DQN_algorithm():

    def __init__(self, agent, node_embedder, env, n_batches,
                 replay_buffer, embedding_dim, T, device,
                 do_neighborhood_embedding=False,
                 use_kernel_net=False,
                 gamma=0.99, max_grad_norm=10.,
                 learning_rate=5e-4, momentum=0.05):

        self.agent = agent
        self.node_embedder = node_embedder
        self.env = env
        self.n_batches = n_batches
        self.embedding_dim = embedding_dim
        self.T = T
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.device = device
        self.optimizer = torch.optim.RMSprop(
            list(self.agent.q_train.parameters()) + 
            list(self.node_embedder.parameters()), lr=learning_rate,
            momentum=momentum
        )
        self.max_grad_norm = max_grad_norm
        self.do_neighborhood_embedding = do_neighborhood_embedding
        self.use_kernel_net = use_kernel_net

    def get_samples_full_episode(self):

        samples = {}

        draw, network_sample = self.replay_buffer.draw_sample()
        for agent in draw:
            samples[agent] = {
                'observation' : [],
                'action' : [],
                'reward' : [],
                'done' : []
            }
            for t in range(len(draw[agent])):
                samples[agent]['observation'].append(draw[agent][t]['observation'])
                samples[agent]['action'].append(draw[agent][t]['action'])
                samples[agent]['reward'].append(draw[agent][t]['reward'])
                samples[agent]['done'].append(draw[agent][t]['done'])

            samples[agent]['observation'] = torch.cat(samples[agent]['observation'])
            samples[agent]['action'] = torch.stack(samples[agent]['action'])
            samples[agent]['reward'] = torch.stack(samples[agent]['reward'])
            samples[agent]['done'] = torch.stack(samples[agent]['done'])

        return samples, network_sample

    def get_samples_full_episode_kernel(self):

        samples = {}

        draw, network_sample = self.replay_buffer.draw_sample()
        for agent in draw:
            samples[agent] = {
                'observation' : [],
                'action' : [],
                'reward' : [],
                'done' : []
            }
            for t in range(len(draw[agent])):
                samples[agent]['observation'].append(draw[agent][t]['observation'][0])
                samples[agent]['action'].append(draw[agent][t]['action'])
                samples[agent]['reward'].append(draw[agent][t]['reward'])
                samples[agent]['done'].append(draw[agent][t]['done'])

            samples[agent]['action'] = torch.stack(samples[agent]['action'])
            samples[agent]['reward'] = torch.stack(samples[agent]['reward'])
            samples[agent]['done'] = torch.stack(samples[agent]['done'])

        return samples, network_sample

    def optimize_agent(self):

        self.optimizer.zero_grad()
        loss, td_abs_errors, qv_train_record, qv_target_record = self.loss()
        loss.backward()
        
        grad_norm_qv = torch.nn.utils.clip_grad_norm_(
            self.agent.q_train.parameters(), self.max_grad_norm
            )
        grad_norm_ne = torch.nn.utils.clip_grad_norm_(
            self.node_embedder.parameters(), self.max_grad_norm
            )


        self.optimizer.step()


        grad_norm = np.array([grad_norm_qv, grad_norm_ne])

        return td_abs_errors, qv_train_record, qv_target_record, grad_norm

    def loss(self):
        losses = 0
        abs_delta = 0
        n = 0

        if not self.use_kernel_net:
            samples, network_sample = self.get_samples_full_episode()
        else:
            samples, network_sample = self.get_samples_full_episode_kernel()

        node_embeddings = self.node_embedder(
            network_sample['graph'], features=network_sample['net_features']
        ).reshape((self.n_batches, -1, self.embedding_dim))

        neighborhood_embedding_by_agents = make_neighborhood_embedding(
            node_embeddings, network_sample['neighborhoods'], self.env.max_degree,
            add_neighbors=self.do_neighborhood_embedding
        )

        for agent in samples:

            observation_node_embedding = cat_obs_node_embedding(
                samples[agent]['observation'].to(self.device),
                neighborhood_embedding_by_agents[agent], self.n_batches, n_time_steps=self.T,
                use_kernel_net=self.use_kernel_net, node_embeddings=node_embeddings
            )
            
            qvals, _ = self.agent.q_train.forward(
                observation_node_embedding
            )
            qval = select_at_indexes(
                samples[agent]['action'],
                qvals)

            with torch.no_grad():
                observation_node_embedding = cat_obs_node_embedding(
                    samples[agent]['observation'].to(self.device),
                    neighborhood_embedding_by_agents[agent], self.n_batches, n_time_steps=self.T,
                    use_kernel_net=self.use_kernel_net, node_embeddings=node_embeddings
                )
                target_qs, _ = self.agent.q_target.forward(
                    observation_node_embedding
                )
                target_q = torch.max(target_qs, dim=-1).values
                y = (
                    samples[agent]['reward'].to(self.device) + 
                    self.gamma * torch.cat([target_q[1:, :], torch.zeros(1, target_q.size(1)).to(self.device)])
                )
            # pdb.set_trace()
            delta = (y - qval)
            delta2 = torch.mean(delta**2)
            losses += 0.5 * delta2 ** 2
            abs_delta += torch.mean(abs(delta.detach()))
            n += 1

        abs_delta = abs_delta / n
        losses = losses / n

        # return the last agent's q-values for a single episode (in the first time step)
        # deprecated - keep only for possible debugging purposes. 
        qv_train_record = qvals.detach().cpu().numpy()[0, 0, :]
        qv_target_record = target_qs.detach().cpu().numpy()[0, 0, :]

        # compute batch averages of min and max q-values for first time step
        min_av_train = qvals.detach().cpu().numpy()[0, :, :].min(axis=1).mean()
        max_av_train = qvals.detach().cpu().numpy()[0, :, :].max(axis=1).mean()

        min_av_target = target_qs.detach().cpu().numpy()[0, :, :].min(axis=1).mean()
        max_av_target = target_qs.detach().cpu().numpy()[0, :, :].max(axis=1).mean()

        # overwrite q-vals (since min, max averages is really what we want)
        qv_train_record = np.array([min_av_train, max_av_train])
        qv_target_record = np.array([min_av_target, max_av_target])

        return losses, abs_delta.cpu().numpy(), qv_train_record, qv_target_record

