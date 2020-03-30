

from multiprocessing import Process
import multiprocessing as mult
import torch
import numpy as np
import pdb

class Runner():

    def __init__(
        self, buffer_by_type, eval_buffer_by_type, 
        algo_by_type, node_embedder,
        train_sampler, eval_sampler, logger, 
        print_every=200,
        n_iterations=1000, reset_target_every=200,
        checkpoint_every=500, eval_every=500,
        seed=1, n_parallel=1, n_batches_ckpt=256
    ):

        self.buffer_by_type = buffer_by_type
        self.eval_buffer_by_type = eval_buffer_by_type
        self.algo_by_type = algo_by_type
        self.node_embedder = node_embedder

        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.logger = logger
        self.n_iterations = n_iterations
        self.reset_target_every = reset_target_every
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.print_every = print_every
        self.seed = seed
        self.n_parallel = n_parallel
        self.n_batches_ckpt = n_batches_ckpt

    def train_process(self, seed):

        torch.manual_seed(seed)
        np.random.seed(seed)

        for agent in self.algo_by_type:
            self.algo_by_type[agent].agent.set_train()

        for its in range(self.n_iterations):
            self.train_sampler.run_episode(
                self.buffer_by_type, its
            )

            for agent in self.algo_by_type:
                td_abs_errors, qv_train_record, qv_target_record, grad_norm = self.algo_by_type[agent].optimize_agent()
                train_reward, _, _, _, _ = self.buffer_by_type[agent].compute_average_reward()
                self.logger.log_train_by_agent(
                    agent, train_reward, td_abs_errors,
                    qv_train_record, qv_target_record, grad_norm
                )
                self.algo_by_type[agent].agent.reset_rnn_state()
                if its % self.reset_target_every == 0:
                    print(50 * '=+')
                    print('Resetting target weights')
                    print(50 * '=+')
                    self.algo_by_type[agent].agent.update_target()
                self.buffer_by_type[agent].reset_buffer()
            
            if its % self.eval_every == 0:
                eval_reward, sd_error, _, _, _ = self.eval(its)
                print(eval_reward, sd_error)

            if its % self.print_every == 0:
                self.logger.print_to_terminal(its)

            if (its + 1) % self.checkpoint_every == 0:
                eval_reward, sd_error, _, _, _ = self.eval(its, log=False, ckpt=True)
                self.logger.save_checkpoint(its, self.algo_by_type, self.node_embedder, eval_reward)

    def eval(self, its, log=True, ckpt=False):

        for agent in self.algo_by_type:
            self.algo_by_type[agent].agent.set_eval()
        
        if ckpt:
            n_batches_prev = self.eval_sampler.env.n_batches
            self.eval_sampler.env.n_batches = self.n_batches_ckpt

        self.eval_sampler.run_episode(
            self.eval_buffer_by_type, its
        )

        eval_rewards = {}
        sd_errors = {}
        frwf = {}
        rewards_quantiles = {}
        degree_acc_corr = {}
                        
        for agent in self.eval_buffer_by_type:
            self.algo_by_type[agent].agent.reset_rnn_state()
            eval_reward, sd_error, frac_wrong_frac, quantiles, deac = self.eval_buffer_by_type[agent].compute_average_reward()
            if log:
                self.logger.log_eval_by_agent(agent, eval_reward, sd_error)
            self.eval_buffer_by_type[agent].reset_buffer()
            self.algo_by_type[agent].agent.set_train()
            eval_rewards[agent] = eval_reward
            sd_errors[agent] = sd_error
            frwf[agent] = frac_wrong_frac
            rewards_quantiles[agent] = quantiles
            degree_acc_corr[agent] = deac

        if ckpt:
            self.eval_sampler.env.n_batches = n_batches_prev

        return eval_rewards, sd_errors, frwf, rewards_quantiles, degree_acc_corr


    def train(self, parallel=False):
        
        if not parallel:
            self.train_process(self.seed)
        else:
            processes = []

            self.n_parallel = min(self.n_parallel, max(mult.cpu_count() - 1, 1))
            print('Running a total of %d experiments in parallel.' % self.n_parallel)

            for e in range(self.n_parallel):
                seed = self.seed + 10*e
                print('Running experiment with seed %d'%seed)

                def train_func():
                    self.train_process(seed)
       
                p = Process(target=train_func, args=tuple())
                p.start()
                processes.append(p)
                # if you comment in the line below, then the loop will block 
                # until this process finishes
                # p.join()

            for p in processes:
                p.join()