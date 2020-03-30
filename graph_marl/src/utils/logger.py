# What to log?
# - model parameter snapshots
# - average terminal rewards by agent_type, training and testing
# - td error by agent_type
# - game parameters
# - gradient diagnostics

import pdb
import torch
import os
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger():

    def __init__(self, save_path, average_window, run_id, experiment_name, tf_board_writer_path):

        self.save_path = save_path
        self.average_window = average_window
        self.run_id = run_id
        self.writer = SummaryWriter(tf_board_writer_path + '/' + experiment_name + '_' + run_id)

        self.training_rewards = {}
        self.eval_rewards = {}
        self.eval_rewards_sd = {}

        self.td_erros = {}
        self.qvals_train = {}
        self.qvals_target = {}
        self.gradnorm = {}

    def log_train_by_agent(
        self, agent, reward, td_error, qvals_train, qvals_target,
        gradnorm
    ):
        if agent not in self.training_rewards:

            self.training_rewards[agent] = []

            self.td_erros[agent] = []
            self.qvals_train[agent] = []
            self.qvals_target[agent] = []
            self.gradnorm[agent] = []

        # only store terminal reward here
        self.training_rewards[agent].append(reward[-1])

        self.td_erros[agent].append(td_error)
       
        self.qvals_train[agent].append(qvals_train)
        self.qvals_target[agent].append(qvals_target)
        self.gradnorm[agent].append(gradnorm)

    def log_eval_by_agent(self, agent, reward, reward_sd):

        if agent not in self.eval_rewards:
            self.eval_rewards[agent] = []
            self.eval_rewards_sd[agent] = []

        # only store terminal reward here
        self.eval_rewards[agent].append(reward[-1])
        self.eval_rewards_sd[agent].append(reward_sd[-1])

    def print_to_terminal(self, iteration):
        
        print(50*'=')
        print('Iteration: %d' % iteration)

        for agent in self.training_rewards:
            print(50*'=')
            print('Agent: %s' % agent)
            print('Training terminal reward = %f' % self.training_rewards[agent][-1])
            self.writer.add_scalar(
                agent + '/training_terminal_reward',
                self.training_rewards[agent][-1], iteration
            )
            if len(self.eval_rewards[agent]) > 0:
                print('Eval terminal reward = %f' % self.eval_rewards[agent][-1])
                print('Eval terminal reward sd (estimate) = %f' % self.eval_rewards_sd[agent][-1])
                self.writer.add_scalar(
                    agent + '/eval_terminal_reward',
                    self.eval_rewards[agent][-1], iteration
                )
            print('Train q values (min, max)= %s' % self.qvals_train[agent][-1])
            print('Target q values (min, max)= %s' % self.qvals_target[agent][-1])

            self.writer.add_scalar(
                    agent + '/train_q_vals_min',
                    self.qvals_train[agent][-1][0], iteration
                )

            self.writer.add_scalar(
                    agent + '/train_q_vals_max',
                    self.qvals_train[agent][-1][1], iteration
                )

            self.writer.add_scalar(
                    agent + '/target_q_vals_min',
                    self.qvals_target[agent][-1][0], iteration
                )

            self.writer.add_scalar(
                    agent + '/target_q_vals_max',
                    self.qvals_target[agent][-1][1], iteration
                )

            if len(self.qvals_target[agent]) >= 300:
                qvt = np.array(self.qvals_train[agent])[-300:, :]
            else:
                qvt = np.array(self.qvals_train[agent])
            print('Train q values mean (300 episode moving window = %s)' % qvt.mean(axis=0))
            print('Train q values sd (300 episode moving window = %s)' % qvt.std(axis=0))
            
            print('Grad norm = %s' % self.gradnorm[agent][-1])
            print('TD error = %s' % self.td_erros[agent][-1])

            self.writer.add_scalar(
                    agent + '/grad_norm_qvals',
                    self.gradnorm[agent][-1][0], iteration
                )

            self.writer.add_scalar(
                    agent + '/grad_norm_node_embedder',
                    self.gradnorm[agent][-1][1], iteration
                )
            
            self.writer.add_scalar(
                    agent + '/td_error',
                    self.td_erros[agent][-1], iteration
                )

            print(50*'=')
            

    def save_checkpoint(self, iteration, algo_by_type, node_embedder, eval_rewards):

        save_dict = {}
        for agent in algo_by_type:
            models = algo_by_type[agent].agent.get_models()
            for m in models:
                save_dict['model_' +  m + '_' + agent] = models[m].state_dict()
            save_dict['optimizer_' + agent] = algo_by_type[agent].optimizer.state_dict()

        save_dict['node_embedder'] = node_embedder.state_dict()

        save_dict['iteration'] = iteration

        reward_str = '_'
        for agent in eval_rewards:
            reward_str += agent + '_rw_' + str(eval_rewards[agent][-1])
        
        chpt_filename = self.run_id + '_iteration_' + str(iteration) + reward_str + '_.tar'
        save_path = os.path.join(self.save_path, 'checkpoints/', self.run_id)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(
            save_dict,
            os.path.join(save_path, chpt_filename)
        )

        log_filename = self.run_id + 'log.pkl'

        with open(os.path.join(save_path, log_filename), 'wb')  as f:
            pickle.dump([
                self.training_rewards, self.eval_rewards, self.td_erros, 
                self.qvals_train, self.qvals_target, self.gradnorm
                ], f, pickle.HIGHEST_PROTOCOL)


        