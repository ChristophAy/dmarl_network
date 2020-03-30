from graph_marl.src.environments.env_network import env_multi
from graph_marl.src.agents.q_agent_network import RQ_agent
from graph_marl.src.running.buffer_network import ReplayBuffer
from graph_marl.src.running.sampler_network import Sampler
from graph_marl.src.running.runner_network import Runner
from graph_marl.src.rl_algo.dqn_algorithm_network import DQN_algorithm
from graph_marl.src.utils.logger import Logger
from graph_marl.src.utils.helper_funs import id_generator
from graph_marl.src.models.graph_model import NodeEmbedding, IdentityEmedding, TypeEmedding
from graph_marl.src.running.de_groot import DeGrootBenchmark
from graph_marl.src.models.graph_model import DummyEmedding

import argparse
# import git
import os
import torch
import numpy as np
import time
import pdb
from copy import deepcopy
from scipy.stats import norm
import cProfile
import pickle
import git

parser = argparse.ArgumentParser()

# NETWORK ARCHITECTURE PARAMETERS
parser.add_argument('-n_hidden', help='GRU and fully connected hidden units', type=int, default=64)
parser.add_argument('-n_layers', help='number of GRU layers', type=int, default=2)
parser.add_argument('-input_dim_node_embedding', help='input dimension of node embedder', type=int, default=3)
parser.add_argument('-hidden_dim_node_embedding', help='input dimension of node embedder', type=int, default=10)
parser.add_argument('-do_neighborhood_embedding', help='whether or not to feed in node embedding of entire neighborhood', action="store_true")
parser.add_argument('-use_attention_net', help='whether to use attention architecture', action="store_true")
parser.add_argument('-use_dummy_embedding',
help='whether to use a dummy embedder, otherwise will train node embedder', action="store_true")
parser.add_argument('-use_identity_embedding',
help='whether to use a identity embedder - just returns node degree otherwise will train node embedder', action="store_true")
parser.add_argument('-use_type_embedding',
help='whether to use a type embedder - just returns node type otherwise will train node embedder', action="store_true")
parser.add_argument('-attention_weight_mode', help='which net to use to compute attention weights', type=str, default='dynamic') 
parser.add_argument('-attention_weight_features', help='which features to use to compute attention', type=str, default='degree') 
parser.add_argument('-use_kernel_net',
help='whether to use the kernel net architecture', action="store_true")


# LEARNING PARAMETERS
parser.add_argument('-seed', help='global seed', type=int, default=12)
parser.add_argument('-learning_rate', help='learning rate', type=float, default=5e-4)
parser.add_argument('-learning_rate_attack', help='learning rate', type=float, default=5e-4)
parser.add_argument('-momentum', help='momentum for RMSProp', type=float, default=0.05) 
parser.add_argument('-optimizer', help='which optimizer to use', type=str, default='RMSProp') 
parser.add_argument('-exploration_rate', help='exploration rate', type=float, default=0.05)
parser.add_argument('-exploration_schedule', help='exploration schedule to use',
type=str, default='constant')
parser.add_argument('-grad_norm_clipping', help='max gradient norm to clip to', type=float, default=10.)
parser.add_argument('-target_reset', help='dqn reset frequency', type=int, default=100)
parser.add_argument('-n_iterations', help='number of training episodes', type=int, default=5000)
parser.add_argument('-n_batches', help='number of batches', type=int, default=64)
parser.add_argument('-n_batches_test', help='number of batches for testing during training',
type=int, default=64)
parser.add_argument('-n_batches_evaluation',
help='number of batches for testing during evaluation post traing and checkpointing',type=int, default=512)

parser.add_argument('-buffer_size', help='size of replay buffer', type=int, default=20)
parser.add_argument('-sample_last',
help='whether to sample last episode, otherwise will sample random episode', action="store_true")

parser.add_argument('-embedding_dims', help='dimension of node embedding', type=int, default=2)
parser.add_argument('-un_normalized_record_rewards',
help='whether NOT to normalize rewards by Bayes optimal benchmark', action="store_true")
parser.add_argument('-use_pre_trained_embedding',
help='whether to use a pre-trained embedding for nodes', action="store_true")
parser.add_argument('-pre_trained_fname', help='file that stores the pre-trained embedder', type=str,
default='pre_trained_node_embedder_.tar') 

parser.add_argument('-n_experiments', help='number of experiments to run in parallel', type=int, default=1)
parser.add_argument('-no_batch_networks', help='randomize networks within batch', action="store_true")
parser.add_argument('-boltzmann_actions',
help='whether to use boltzmann action selection', action="store_true")
parser.add_argument('-boltzmann_temp', help='boltzmann temperature', type=float, default=0.)
parser.add_argument('-use_cpu', help='whether to force use of cpu', action="store_true")

# GAME PARAMETERS
parser.add_argument('-num_actions', help='number of actions', type=int, default=2)
parser.add_argument('-n_states', help='number of states', type=int, default=2)
parser.add_argument('-max_agents', help='max number of agents', type=int, default=20)
parser.add_argument('-min_agents', help='min number of agents', type=int, default=20)
parser.add_argument('-max_degree', help='max number of neighbors', type=int, default=20)
parser.add_argument('-var', help='variance of signal', type=float, default=1.)
parser.add_argument('-T', help='number of time steps', type=int, default=10)
parser.add_argument('-gamma', help='discount factor', type=float, default=0.99)
parser.add_argument('-network_path', help='location of network file', default='social_network_files/')
parser.add_argument('-network_file', help='network file', default='social_network_complete.txt')
parser.add_argument('-network_type', help='type of network generator', default='facebook')
parser.add_argument('-attack', help='whether to run in attack mode', action="store_true")
parser.add_argument('-n_attack', help='number of nodes to attack', type=int, default=0)
parser.add_argument('-bias_per_agent', help='attack bias per agent', type=float, default=0.)
parser.add_argument('-random_attacker', help='if random attacker should be used instead of trained attacker', action="store_true")
parser.add_argument('-trained_attacker', help='if trained attacker should be used', action="store_true")
parser.add_argument('-signal_var_types', help='if there should be agents with high and low signal precision', action="store_true")
parser.add_argument('-type_label', help='label used for types', default='hl_label')
parser.add_argument('-n_h_type', help='number of nodes of h type', type=int, default=0)
parser.add_argument('-low_precision_var_factor', help='factor by which to increase signal variance of low precision signal agents', type=float, default=0.)
parser.add_argument('-n_core', help='number of nodes in core', type=int, default=1)
parser.add_argument('-attack_full_action_obs', help='if attacker can observe all actions', action="store_true")
parser.add_argument('-attack_full_signal_obs', help='if attacker can observe all signals', action="store_true")
parser.add_argument('-fixed_node_encoding', help='if each node receives a unique code', action="store_true")

# LOGGING PARAMTERS
parser.add_argument('-print_every', help='reward printing interval', type=int, default=10)
parser.add_argument('-average_window', help='print averaging interval', type=int, default=300)
parser.add_argument('-save_path', help='location of model checkpoints', default='checkpoints/')
parser.add_argument('-code_version', help='git version of code', default='no_version')
parser.add_argument('-experiment_name', help='name of experiment for easy identification', default='no_name')
parser.add_argument('-checkpt_freq', help='frequency of model checkpoints under regular saving', type=int, default=100)
parser.add_argument('-eval_every', help='frequency to run evaluation', type=int, default=10)

# RESTORE AND EVALUATE PARAMETERS
parser.add_argument('-restore', help='restore best saved model', action="store_true")
parser.add_argument('-restore_set_params', help='if to reset some of the parameters to custom', action="store_true")
parser.add_argument('-restore_file', help='file of run to restore', type=str, default='')
parser.add_argument('-restore_run_id', help='run id of run to restore', type=str, default='')
parser.add_argument('-restore_best_min_it',
help='minimum number of iterations a checkpoint should have to be loaded under restore best mode', type=int, default=2000)
parser.add_argument('-evaluate', help='evaluate a saved model', action="store_true")
parser.add_argument('-degroot_only', help='evaluate only de groot model', action="store_true")
parser.add_argument('-save_path_evals', help='where to store evaluation results', type=str, default='')
parser.add_argument('-save_fname_prefix', help='prefix to add to savename file', type=str, default='')

args = parser.parse_args()

if args.restore:

    restore_path = os.path.abspath('results/' + args.experiment_name + '/checkpoints/' + args.restore_run_id + '/')
    with open(os.path.join(restore_path, 'args_'+ args.restore_run_id), 'rb') as f:
        args_restore = pickle.load(f)[0]
    if 'attack_full_action_obs' not in args_restore:
        args_restore.attack_full_action_obs = False
    if 'attack_full_signal_obs' not in args_restore:
        args_restore.attack_full_signal_obs = False
    if 'fixed_node_encoding' not in args_restore:
        args_restore.fixed_node_encoding = False
    if 'use_cpu' not in args_restore:
        args_restore.use_cpu = False
    if 'boltzmann_actions' not in args_restore:
        args_restore.boltzmann_actions = False
    if 'boltzmann_temp' not in args_restore:
        args_restore.boltzmann_temp = 0.
    
    if args.restore_set_params:
        # args_restore.T = args.T
        args_restore.max_agents = args.max_agents
        args_restore.min_agents = args.min_agents
        # args_restore.max_degree = args.max_degree
        # args_restore.var = args.var
        # args_restore.attack = args.attack
        args_restore.n_attack = args.n_attack
        # args_restore.bias_per_agent = args.bias_per_agent
        args_restore.un_normalized_record_rewards = args.un_normalized_record_rewards
        args_restore.network_type = args.network_type
        args_restore.network_file = args.network_file
        args_restore.network_path = args.network_path
        
    args_restore.seed = args.seed
    args_restore.n_batches_test = args.n_batches_test
    args_restore.n_batches_evaluation = args.n_batches_evaluation
    args_restore.restore = args.restore
    args_restore.restore_run_id = args.restore_run_id
    args_restore.experiment_name = args.experiment_name
    args_restore.evaluate = args.evaluate
    args_restore.degroot_only = args.degroot_only
    args_restore.save_path_evals = args.save_path_evals
    args_restore.save_fname_prefix = args.save_fname_prefix

    if args.restore_file == 'highest_accuracy':
        all_files = [f for f in os.listdir(restore_path) if 'iteration' in f]
        best_file = all_files[-1]
        best_accuracy = float(best_file.split('_')[5].strip('attacker'))
        for f in all_files:
            acc = float(f.split('_')[5].strip('attacker'))
            if int(f.split('_')[2]) > args.restore_best_min_it and acc > best_accuracy:
                best_accuracy = acc
                best_file = f
        args_restore.restore_file = best_file
    else:
        args_restore.restore_file = args.restore_file

    print(100*'=')
    print('Will restore checkpoint : %s' % args_restore.restore_file)
    print(100*'=')

    args = deepcopy(args_restore)

args.device = None
if not args.use_cpu and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print(args.device)

print(args.max_agents)

if args.use_identity_embedding:
    args.embedding_dims = 1
if args.use_type_embedding:
    args.embedding_dims = 1

args_eval = deepcopy(args)
args_eval.buffer_size = 1
args_eval.n_batches = args_eval.n_batches_test

# add code version number

# repo = git.Repo(search_parent_directories=True)
# args.code_version = repo.head.object.hexsha

# Only uncomment this when running on server - in fact it's fine if left
torch.set_num_threads(10)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

# Create an environment

USE_NODE_EMBEDDING = not args.use_dummy_embedding
NORMALIZE_RECORD_REWARDS = not args.un_normalized_record_rewards

if USE_NODE_EMBEDDING:
    if args.use_identity_embedding:
        node_embedder = IdentityEmedding(args.input_dim_node_embedding,
                                  args.hidden_dim_node_embedding, args.embedding_dims)
    elif args.use_type_embedding:
        node_embedder = TypeEmedding(args.input_dim_node_embedding,
                                  args.hidden_dim_node_embedding, args.type_label, args.embedding_dims)
    else:
        node_embedder = NodeEmbedding(args.input_dim_node_embedding,
                                    args.hidden_dim_node_embedding, args.device, embedding_dim=args.embedding_dims)
    if args.use_pre_trained_embedding:
        save_path_ne = os.path.abspath('results')
        filename_ne = args.pre_trained_fname
        save_path_ne = os.path.join(save_path_ne, 'pre_trained_node_embedder/checkpoints/', filename_ne)
        checkpoint_ne = torch.load(save_path_ne, map_location=args.device)
        # [print(p) for p in node_embedder.parameters()]
        node_embedder.load_state_dict(checkpoint_ne['node_embedder'])
        # print(50*'=')
        # [print(p) for p in node_embedder.parameters()]
else:
    node_embedder = DummyEmedding(args.input_dim_node_embedding,
                                  args.hidden_dim_node_embedding, args.embedding_dims)

node_embedder.to(args.device)

if args.do_neighborhood_embedding:
    obs_dim = args.max_degree * (args.num_actions + 1) + args.embedding_dims * (args.max_degree + 1) + 1
    obs_dim += args.T
elif args.fixed_node_encoding:
    obs_dim = args.max_degree * (args.num_actions + 1) + args.embedding_dims + 1
    obs_dim += args.T
    obs_dim += args.max_agents
else:
    obs_dim = args.max_degree * (args.num_actions + 1) + args.embedding_dims + 1
    obs_dim += args.T


if args.fixed_node_encoding:
    obs_idx = (
        args.max_degree * (args.num_actions + 1) + 1 + args.T + args.max_agents,
        args.max_degree * (args.num_actions + 1) + 1 + args.max_degree + args.T + args.max_agents,
        args.max_degree * (args.num_actions + 1) + args.max_agents,
        args.max_degree * (args.num_actions + 1) + 1 + args.max_agents,
    )
else:
    obs_idx = (
        args.max_degree * (args.num_actions + 1) + 1 + args.T,
        args.max_degree * (args.num_actions + 1) + 1 + args.max_degree + args.T,
        args.max_degree * (args.num_actions + 1),
        args.max_degree * (args.num_actions + 1) + 1,
    )

q_agent = RQ_agent(args.max_agents, obs_dim, args.n_hidden, args.n_layers, 
                    args.num_actions, args.max_degree, args.T, args.exploration_rate,
                    args.device,
                    use_attention_net=args.use_attention_net, 
                    use_kernel_net=args.use_kernel_net,
                    node_embedding_dim=args.embedding_dims,
                    obs_idx=obs_idx,
                    attention_weight_mode=args.attention_weight_mode,
                    attention_weight_features=args.attention_weight_features,
                    boltzmann=args.boltzmann_actions, temperature=args.boltzmann_temp
                    )
agent_type_dict = {'citizen' : q_agent}

if args.trained_attacker:

    if args.fixed_node_encoding:
        print('Fixed node encoding not support under trained attack')
        raise NotImplementedError()

    signal_n_obs = 1
    action_n_obs = args.max_degree

    if args.attack_full_signal_obs:
        signal_n_obs = args.max_agents
    if args.attack_full_action_obs:
        action_n_obs = args.max_agents

    if args.do_neighborhood_embedding:
        obs_dim = action_n_obs * (args.num_actions + 1) + args.embedding_dims * (args.max_degree + 1) + signal_n_obs
        obs_dim += args.T
    else:
        obs_dim = action_n_obs * (args.num_actions + 1) + args.embedding_dims + signal_n_obs
        obs_dim += args.T

    obs_idx = (
        action_n_obs * (args.num_actions + 1) + signal_n_obs + args.T,
        action_n_obs * (args.num_actions + 1) + signal_n_obs + action_n_obs + args.T,
        action_n_obs * (args.num_actions + 1),
        action_n_obs * (args.num_actions + 1) + signal_n_obs
    )

    q_agent_attacker = RQ_agent(args.max_agents, obs_dim, args.n_hidden, args.n_layers, 
                    args.num_actions, args.max_degree, args.T, args.exploration_rate,
                    args.device,
                    use_attention_net=args.use_attention_net, 
                    use_kernel_net=args.use_kernel_net,
                    node_embedding_dim=args.embedding_dims,
                    obs_idx=obs_idx, signal_dim=signal_n_obs,
                    full_act_obs=action_n_obs,
                    attention_weight_mode=args.attention_weight_mode,
                    attention_weight_features=args.attention_weight_features,
                    boltzmann=args.boltzmann_actions, temperature=args.boltzmann_temp
                    )
    agent_type_dict['attacker'] = q_agent_attacker


environment = env_multi(node_embedder, agent_type_dict, args)
environment_eval = env_multi(node_embedder,agent_type_dict, args_eval)

random_sample = not args.sample_last

replay_buffer = ReplayBuffer(buffer_size=args.buffer_size, random_sample=random_sample,
                             reward_rel_benchmark=NORMALIZE_RECORD_REWARDS, signal_var=args.var)
replay_buffer_eval = ReplayBuffer(buffer_size=args_eval.buffer_size, random_sample=random_sample,
                                  reward_rel_benchmark=NORMALIZE_RECORD_REWARDS, signal_var=args_eval.var)

buffer_by_type = {'citizen' : replay_buffer}
eval_buffer_by_type = {'citizen' : replay_buffer_eval}

if args.trained_attacker:

    replay_buffer_attacker = ReplayBuffer(buffer_size=args.buffer_size, random_sample=random_sample,
                             reward_rel_benchmark=NORMALIZE_RECORD_REWARDS, signal_var=args.var)
    replay_buffer_eval_attacker = ReplayBuffer(buffer_size=args_eval.buffer_size, random_sample=random_sample,
                                  reward_rel_benchmark=NORMALIZE_RECORD_REWARDS, signal_var=args_eval.var)
    
    buffer_by_type['attacker'] = replay_buffer_attacker
    eval_buffer_by_type['attacker'] = replay_buffer_eval_attacker

train_sampler = Sampler(environment, args)
eval_sampler = Sampler(environment_eval, args_eval, is_eval=True)

dqn_algo = DQN_algorithm(q_agent, node_embedder, environment, args.n_batches,
                        replay_buffer, args.embedding_dims, args.T, args.device,
                        do_neighborhood_embedding=args.do_neighborhood_embedding,
                        use_kernel_net=args.use_kernel_net,
                        gamma=args.gamma, max_grad_norm=args.grad_norm_clipping,
                        learning_rate=args.learning_rate, momentum=args.momentum)

algo_by_type = {'citizen' : dqn_algo}

if args.trained_attacker:

    dqn_algo_attacker = DQN_algorithm(q_agent_attacker, node_embedder, environment, args.n_batches,
                        replay_buffer_attacker, args.embedding_dims, args.T, args.device,
                        do_neighborhood_embedding=args.do_neighborhood_embedding,
                        use_kernel_net=args.use_kernel_net,
                        gamma=args.gamma, max_grad_norm=args.grad_norm_clipping,
                        learning_rate=args.learning_rate_attack, momentum=args.momentum)
    algo_by_type['attacker'] = dqn_algo_attacker


experiment_name = args.experiment_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
save_path = os.path.abspath(os.path.join('results', experiment_name))
run_id = id_generator()

if args.restore:
    restore_path = os.path.abspath('results/' + args.experiment_name + '/checkpoints/' + args.restore_run_id + '/')
    fname = args.restore_file
    checkpoint = torch.load(os.path.join(restore_path, fname), map_location=args.device)
    for agent in algo_by_type:
        algo_by_type[agent].agent.q_train.load_state_dict(checkpoint['model_' +  'q_train' + '_' + agent])
        algo_by_type[agent].agent.q_target.load_state_dict(checkpoint['model_' +  'q_target' + '_' + agent])
        algo_by_type[agent].optimizer.load_state_dict(checkpoint['optimizer_' + agent])
    # [print(p) for p in node_embedder.parameters()]
    node_embedder.load_state_dict(checkpoint['node_embedder'])
    # [print(p) for p in node_embedder.parameters()]
else:
    path_tmp = os.path.join(save_path, 'checkpoints/', run_id)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    with open(os.path.join(path_tmp, 'args_' + run_id), 'wb')  as f:
            pickle.dump([
                args
                ], f, pickle.HIGHEST_PROTOCOL)

tf_board_writer_path =os.path.abspath('results/runs')
my_logger = Logger(save_path, args.average_window, run_id, args.experiment_name, tf_board_writer_path)

print('seed = %d' % args.seed)

my_runner = Runner(
    buffer_by_type, eval_buffer_by_type, 
    algo_by_type, node_embedder,
    train_sampler, eval_sampler, my_logger,
    n_iterations=args.n_iterations,
    reset_target_every=args.target_reset,
    checkpoint_every=args.checkpt_freq,
    eval_every=args.eval_every,
    print_every=args.print_every,
    seed=args.seed, n_parallel=args.n_experiments,
    n_batches_ckpt=args.n_batches_evaluation
)

if args.evaluate:
    dgb = DeGrootBenchmark(environment_eval, args_eval.T, NORMALIZE_RECORD_REWARDS)
    accuracies_dg, sd_error_dg = dgb.run_episode()
    print(accuracies_dg)

    if not args.degroot_only:

        eval_rewards, sd_errors, frac_wrong_frac, quantiles, degree_acc_corr = my_runner.eval(0, log=False, ckpt=True)
        
        df_dict = {}
        for agent in eval_rewards:
            df_dict[agent + '_m_reward'] = eval_rewards[agent]
            df_dict[agent + '_sde_reward'] = sd_errors[agent]

            if agent == 'citizen':
                fwf = frac_wrong_frac[agent]
                qtl = quantiles[agent]
                deac = degree_acc_corr[agent]
                for f in range(fwf.shape[0]):
                    df_dict[agent + '_fwf_' + str(fwf[f, 0])] = fwf[f, 1]
                for q in range(qtl.shape[0]):
                    df_dict[agent + '_rquantile_' + str(qtl[q, 0])] = qtl[q, 1]
                df_dict[agent + '_degree_acc_corr'] = deac

        df_dict['de_groot' + '_m_reward'] = accuracies_dg
        df_dict['de_groot' + '_sde_reward'] = sd_error_dg

        for arg in vars(args):
            df_dict[arg] = getattr(args, arg)

        print(eval_rewards)
        print(sd_errors)

        import pandas as pd
        df = pd.DataFrame(df_dict)


        if 'attacker' in eval_rewards:
            print(df[['citizen_m_reward', 'attacker_m_reward','de_groot_m_reward', 'citizen_degree_acc_corr']].head())
        else:
            print(df[['citizen_m_reward','de_groot_m_reward']].head())

        save_path_evals = os.path.abspath('results/' + args.save_path_evals)
        evals_fname = args.save_fname_prefix + '_' + args.experiment_name.split('/')[-1] + '_' + args.restore_run_id + '.csv'
        df.to_csv(os.path.join(save_path_evals, evals_fname))

else:
    # cProfile.runctx('my_runner.train(parallel=False)', None, locals())
    my_runner.train(parallel=False)