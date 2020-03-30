import torch
import pdb
from collections import OrderedDict

class Sampler():

    def __init__(self, environment, args, is_eval=False):

        self.env = environment
        self.args = args
        self.is_eval = is_eval

    def initialize_sample_dict(self, agents_by_type):

        samples = OrderedDict()
        for a_type in agents_by_type:
            samples[a_type] = OrderedDict()
            for agent in agents_by_type[a_type]:
                samples[a_type][agent] =  {
                    'observation' : None,
                    'rnn_state' : None,
                    'action' : None,
                    'reward' : None,
                    'done' : None
                }

        return samples

    def get_sample(self, time_step, buffer_by_type, its, observations=None, neighborhood_embedding_by_agents=None,
                   node_embeddings=None):
        
        if time_step == 0:
            observations, neighborhood_embedding_by_agents, node_embeddings = self.env.reset()
            for a_type in buffer_by_type:
                net = {'graph' : self.env.graph, 'neighborhoods' : self.env.neighborhoods,
                       'net_features' : self.env.net_features}
                    #    'attacker_idx' : self.env.attack_idx}
                buffer_by_type[a_type].add_network(net, its)

        samples = self.initialize_sample_dict(self.env.agents_by_type)

        for a_type in self.env.agents_by_type:
            for agent_id, agent in self.env.agents_by_type[a_type].items():
                if not self.is_eval:
                    action, rnn_state, qvals = agent.take_action(agent_id, observations[a_type][agent_id],
                                                                 neighborhood_embedding_by_agents[agent_id], node_embeddings,
                                                                 time_step)
                else:
                    action, rnn_state, qvals = agent.take_action(agent_id, observations[a_type][agent_id],
                                                                 neighborhood_embedding_by_agents[agent_id], node_embeddings,
                                                                 time_step,
                                                                 testing=True)
                samples[a_type][agent_id]['observation'] = observations[a_type][agent_id]
                samples[a_type][agent_id]['rnn_state'] = rnn_state
                samples[a_type][agent_id]['action'] = action
        
        observations, neighborhood_embedding_by_agents, rewards, done = self.env.step(samples)

        for a_type in self.env.agents_by_type:
            for agent in self.env.agents_by_type[a_type]:
                samples[a_type][agent]['reward'] = rewards[a_type][agent]
                samples[a_type][agent]['done'] = done
                buffer_by_type[a_type].add_sample(
                    agent, samples[a_type][agent], its
                )

        return observations, neighborhood_embedding_by_agents, node_embeddings

    def run_episode(self, buffer_by_type, its):
        
        observations = None
        neighborhood_embedding_by_agents = None
        node_embeddings = None
        for time_step in range(self.args.T): 
            observations, neighborhood_embedding_by_agents, node_embeddings = self.get_sample(
                time_step,buffer_by_type, its, observations=observations,
                neighborhood_embedding_by_agents=neighborhood_embedding_by_agents, node_embeddings=node_embeddings
            )