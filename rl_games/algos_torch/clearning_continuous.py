from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd
# from rl_games.algos_torch import central_value, rnd_curiosity
from rl_games.common import vecenv
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import experience

from tensorboardX import SummaryWriter
from datetime import datetime

from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
# TODO: Change all self.env to self.vec_env
from typing import List
# import threading 
# from mpi4py import MPI

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        # self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        # with self.lock:
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]

    # sync the parameters across the cpus
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        # with self.lock:
        local_count = self.local_count.copy()
        local_sum = self.local_sum.copy()
        local_sumsq = self.local_sumsq.copy()
        # reset
        self.local_count[...] = 0
        self.local_sum[...] = 0
        self.local_sumsq[...] = 0
        # synrc the stats
        # sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        sync_sum, sync_sumsq, sync_count = local_sum, local_sumsq, local_count
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # average across the cpu's data
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    # normalize the observation
    def normalize(self, v, clip_range=None):
        # print(v.shape)
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)

class ClearningAgent:
    def __init__(self, base_name, config):
        print(config)
        self.base_init(base_name, config)
        self.target_network_copy_freq = config.get('target_network_copy_freq', 4)
        self.goal_achievement_epsilon = config.get('goal_achievement_epsilon', 0.0)

        self.random_exploration_eps = config.get('random_exploration_eps', 15)
        self.goal_directed_eps = config.get('goal_directed_eps', 100000)
        self.max_ep_len = config.get('max_ep_len', 500)

        self.batch_size = config.get('batch_size', 256)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.learning_rate_critic = config.get('critic_learning_rate', 1e-4)
        self.learning_rate_actor = config.get('actor_learning_rate', 1e-4)

        self.train_steps_per_ep = config.get('train_steps_per_ep', 80)
        self.eval_freq = config.get('eval_freq', 1)
        self.num_eval_goals = config.get('num_eval_goals', 1)
        self.policy_freq = config.get('policy_freq', 2)
        
        self.noise_eps = config.get('noise_eps', 0.3)
        self.exploration_epsilon = config.get('exploration_epsilon', 0.3)
        self.horizon_sampling_const = config.get('horizon_sampling_const', 2.3)
        
        self.use_LR_scheduler = config.get('use_LR_scheduler', True)
        self.LR_change_episode = config.get('LR_change_episode', 2000)
        self.LR_reduce_factor = config.get('LR_reduce_factor', 10)
        self.use_decaying_epsilon = config.get('use_decaying_epsilon', True)
        self.epsilon_decay_denominator = config.get('epsilon_decay_denominator', 150)

        self.use_HER = config.get('use_HER', True)
        self.HER_fraction = config.get('HER_fraction', 0.8)
        self.c_clipping = config.get('c_clipping', False)

        
        self.seed = config.get('overall_seed', 7)
        
        self.random_exploration_eps = config['random_exploration_eps']
        self.max_ep_len = config['max_ep_len']
        self.goal_directed_eps = config['goal_directed_eps']
        self.exploration_epsilon = config['exploration_epsilon']
        self.noise_epsilon = config['noise_eps']

        self.eval_freq = config['eval_freq']

        self.num_eval_goals = config['num_eval_goals']
        self.train_steps_per_ep = config['train_steps_per_ep']
        self.batch_size = config['batch_size']

        # TODO: Might not be needed
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.training_loss = nn.BCELoss()

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape) # TODO: Necessary?
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'goal_dim': 4
            # 'num_seqs' : self.num_actors * self.num_agents
        } 
        self.model = self.network.build(config)
        self.model.to(self.device) # TODO: Get device
        # print(self.model)
        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.optimizer_critic1 = torch.optim.Adam(self.model.clearning_network.critic1.parameters(), lr=self.learning_rate_critic)
        self.optimizer_critic2 = torch.optim.Adam(self.model.clearning_network.critic2.parameters(), lr=self.learning_rate_critic)
        self.optimizer_actor = torch.optim.Adam(self.model.clearning_network.actor.parameters(), lr=self.learning_rate_actor)
        print(self.observation_space)
        self.o_norm = normalizer(size=self.observation_space.shape[0], default_clip_range=5) # TODO
        self.g_norm = normalizer(size=4, default_clip_range=5) # TODO
        # TODO: Algo_Observer?
        self.dataset = []
        self.eval_goal_list = []
        self.eval_mean_success = []
        self.eval_mean_step = []
        self.eval_pts = []
        self.env_steps = []
        self.total_steps_taken = 0
        self.train_step_count = 0

        self.batch_sampling_rng = np.random.RandomState(self.seed + 1)
        self.exploration_rng = np.random.RandomState(self.seed + 2)

    def base_init(self, base_name, config):
        self.config = config
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.device = config.get('device', 'cuda:0')
        print('Env info:')
        print(self.env_info)

        self.observation_space = self.env_info['observation_space']
        self.action_space = self.env_info['action_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.is_train = config.get('is_train', True)

        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        # self.rnn_states = None
        self.name = base_name

        # self.ppo = config['ppo']
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)

        self.obs_shape = self.observation_space.shape


        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.obs = None

        # self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        
        # self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self.device)

    
    def get_full_state_weights(self):
        state = self.get_weights()

        state['steps'] = self.step #TODO
        state['optimizer_actor'] = self.optimizer_actor.state_dict()
        state['optimizer_critic1'] = self.optimizer_critic1.state_dict()
        state['optimizer_critic2'] = self.optimizer_critic2.state_dict()       

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        if self.has_curiosity:
            state['rnd_nets'] = self.rnd_curiosity.state_dict()
        return state

    def get_weights(self):
        state = {'actor': self.model.clearning_network.actor.state_dict(),
         'target_actor' : self.model.clearning_network.target_actor.state_dict(),
         'critic1': self.model.clearning_network.critic1.state_dict(), 
         'target_critic1': self.model.clearning_network.target_critic1.state_dict(),
         'critic2': self.model.clearning_network.critic2.state_dict(), 
         'target_critic2': self.model.clearning_network.target_critic2.state_dict()}

        return state

           

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        if self.has_curiosity:
            state['rnd_nets'] = self.rnd_curiosity.state_dict()
        return state
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_scheckpoint(fn, state)

    def set_weights(self, weights):
        self.model.clearning_network.actor.load_state_dict(weights['actor'])
        self.model.clearning_network.target_actor.load_state_dict(weights['target_actor'])

        self.model.clearning_network.critic1.load_state_dict(weights['critic1'])
        self.model.clearning_network.target_critic1.load_state_dict(weights['target_critic1'])

        self.model.clearning_network.critic2.load_state_dict(weights['critic2'])
        self.model.clearning_network.target_critic2.load_state_dict(weights['target_critic2'])

        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])
        # if self.normalize_reward:
        #     self.reward_mean_std.load_state_dict(weights['reward_mean_std'])

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.step = weights['step']
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])
        if self.has_curiosity:
            self.rnd_curiosity.load_state_dict(weights['rnd_nets'])

        self.optimizer_actor.load_state_dict(weights['optimizer_actor'])
        self.optimizer_critic1.load_state_dict(weights['optimizer_critic1'])
        self.optimizer_critic2.load_state_dict(weights['optimizer_critic2'])

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)


    def preproc_og(self, o):
        # o = np.clip(o, -200, 200)
        return o

    # update parameters in the o_norm normalizer
    def update_o_norm(self, o_norm, episode):
        input = []
        ep_steps = len(episode)
        for i in range(ep_steps):
            input.append(np.array(episode[i]['observation'].cpu().detach().numpy().copy()).squeeze())

        input = self.preproc_og(np.array(input))
        o_norm.update(input)
        o_norm.recompute_stats()

    # update parameters in the g_norm normalizer
    def update_g_norm(self, env, g_norm):
        generated_goals = []
        for _ in range(10000):
            obs = env.reset()
            desired_goal_pose = obs[31:38]
            desired_goal = desired_goal_pose[3:7]
            generated_goals.append(desired_goal.cpu().detach().numpy().copy())
        generated_goals_clip = self.preproc_og(np.array(generated_goals))
        g_norm.update(generated_goals_clip)
        g_norm.recompute_stats()
        
    def run_episode(self, action_policy, ep_length):
        observation = self.vec_env.reset() # TODO
        ep_history = [[]] * self.num_actors
        success = torch.zeros(observation.shape[0], device=self.device)
        for _ in range(ep_length):
            selected_actions = torch.empty(self.num_actors, *self.env_info["action_space"].shape).to(self.device)
            for i in range(self.num_actors):
                desired_goal_pose = observation[i][31:38]
                desired_goal = desired_goal_pose[3:7]
                selected_actions[i]  = torch.from_numpy(action_policy(observation[i].cpu().detach().numpy(), desired_goal.cpu().detach().numpy())).to(self.device)
            new_observation, reward, done, _ = self.vec_env.step(selected_actions) # TODO


            # Store transition in episode buffer
            # Extend it out 
            # Maybe make ep_history another (num_actors, *everything else) array
            for i in range(self.num_actors):
                if not (success[i] == 1 or done[i] == True):
                    achieved_goal_pose = new_observation[i][24:31]
                    desired_goal_pose = new_observation[i][31:38]
                    achieved_goal = achieved_goal_pose[3:7]
                    desired_goal = desired_goal_pose[3:7]
                    ep_history[i].append({"observation": observation[i].clone().detach().cpu().numpy(),
                                    "action": selected_actions[i].clone().detach().cpu().numpy(),
                                    "observation_next": new_observation[i].clone().detach().cpu().numpy(),
                                    "achieved_goal": achieved_goal.clone().detach().cpu().numpy(),
                                    "desired_goal": desired_goal.clone().detach().cpu().numpy()})
            observation = new_observation
            success[reward > -0.5] = 1

        return np.array(ep_history), success.cpu().detach().numpy()

    def goal_conditioned_c_learning_policy(self, rng, exploration_epsilon=0.2, eval=False,
                                           horizon=None, noise_epsilon=0.2, g_norm=None, o_norm=None):

        max_horizon = horizon if horizon is not None else 50
        horizon_vals = np.array([i + 1 for i in range(max_horizon)]).reshape((-1, 1))

        def policy(state, goal):

            # state = o_norm.normalize(state)
            # goal = g_norm.normalize(goal)

            tiled_state = np.tile(state, max_horizon).reshape((max_horizon, -1))
            tiled_goal = np.tile(goal, reps=max_horizon).reshape((max_horizon, -1))

            x_action = torch.cat(
                (torch.tensor(tiled_state).float(), torch.tensor(tiled_goal).float(),
                 torch.tensor(horizon_vals).float()),
                dim=1).to(self.device)

            exploration_p = rng.uniform(low=0.0, high=1.0)

            if exploration_p < exploration_epsilon and not eval:
                a = rng.uniform(-1, 1, size=self.action_space.shape[0])
            else:
                actions = self.model.actor(x_action)
                if eval:
                    action_noise = actions
                else:
                    n = noise_epsilon * np.random.randn(self.action_space.shape[0])
                    tiled_noise = np.tile(n, max_horizon).reshape((max_horizon, -1))
                    tiled_noise = torch.tensor(tiled_noise).float().to(self.device)
                    action_noise = actions + tiled_noise
                    action_noise = torch.clamp(action_noise, self.action_range[0], self.action_range[1]) # Can probably be done with * unzipping

                x = torch.cat((x_action, torch.tensor(action_noise).float()), dim=1)
                accessibilities = self.model.critic1(x).detach().cpu().numpy()
                max_accessibility = accessibilities.max()
                filter_level = 0.9 * max_accessibility
                attainable_horizons = (accessibilities >= filter_level).any(axis=1)
                # argmax only extracts 'True' values, and always returns the first.
                min_attainable_horizon = attainable_horizons.argmax()
                a = action_noise[min_attainable_horizon].detach().cpu().numpy()
            return a

        return policy

    def sample_long_range_transitions(self, dataset: List, batch_size, rng: np.random.RandomState, env,
                                  horizon_sampling_probs: np.array = None, use_HER= True, her_fraction=0.8):


        prob = (np.arange(len(dataset)) + 1)

        ep_idx_sample = rng.choice(len(dataset), size=batch_size, replace=True, p=prob / np.sum(prob))
        ep_sample = [dataset[idx] for idx in ep_idx_sample]
        # print(ep_sample[0])

        start_idx_sample = [
            rng.choice(len(ep)) for ep
            in ep_sample]

        # print(start_idx_sample)

        if horizon_sampling_probs is None:
            h_sample = np.array([rng.choice(len(ep) - start_idx) + 1 for ep, start_idx in zip(ep_sample, start_idx_sample)])
        else:

            max_h = len(horizon_sampling_probs)
            h_sample = np.array([rng.choice(max_h, p=horizon_sampling_probs / horizon_sampling_probs.sum(), size=batch_size)+ 1])


        states_sample = np.empty((batch_size, self.observation_space.shape[0]))
        actions_sample = np.empty((batch_size, self.action_space.shape[0]))
        states_prime_sample = np.empty((batch_size, self.observation_space.shape[0]))
        goals_achieved_sample = np.empty((batch_size, 4))
        h_sample = np.empty((batch_size, 1))
        goals_sample = np.empty((batch_size, 4))

        i = 0
        # print(len(ep_sample), len(start_idx_sample))
        for ep, start_idx in zip(ep_sample, start_idx_sample):
            # print(i)
            states_sample[i] = np.array(ep[start_idx]["observation"])
            actions_sample[i] = np.array(ep[start_idx]["action"])
            states_prime_sample[i] = np.array(ep[start_idx]["observation_next"])
            goals_achieved_sample[i] = np.array(ep[start_idx]["achieved_goal"])
            if use_HER:
                goal_achieved_idx_sample = rng.choice(range(start_idx, len(ep)))
                goal_achieved = np.array(ep[goal_achieved_idx_sample]["achieved_goal"])
                goal_desired = np.array(ep[0]["desired_goal"])
                # print(goal_achieved.shape)
                # print(goal_desired.shape)

                s = rng.choice([0, 1], p=[her_fraction, 1 - her_fraction])
                goals_sample[i] = np.array(goal_achieved if s == 0 else goal_desired)
            else:
                end_s = env.sample_goal_from_state(states_sample[i,:], h_sample[0][i], rng)
                goals_sample[i] = np.array(end_s)
            i += 1
        return torch.from_numpy(states_sample).to(self.device), torch.from_numpy(actions_sample).to(self.device), torch.from_numpy(goals_sample).to(self.device), torch.from_numpy(h_sample).to(self.device), torch.from_numpy(states_prime_sample).to(self.device), torch.from_numpy(goals_achieved_sample).to(self.device)


    
    def learn(self, gde_ep, dataset, batch_size, batch_sampling_rng, o_norm, g_norm):
        # Sample shorter transitions towards the start of training and high transition towards the end
        horizon_sampling_probs = (np.arange(50) + 1) ** -(
                    self.horizon_sampling_const * (1.0 - gde_ep / self.goal_directed_eps))

        # sample minibatch
        states_sample, actions_sample, goals_sample, horizons_sample, s_prime_sample, goal_achieved_sample = \
            self.sample_long_range_transitions(dataset, batch_size=self.batch_size, rng=self.batch_sampling_rng,
                                          horizon_sampling_probs=horizon_sampling_probs, env =self.vec_env,
                                          her_fraction=self.HER_fraction, use_HER=self.use_HER)

        if self.train_step_count % self.target_network_copy_freq == 0:
            self.model.clearning_network.update_targets()

        # save the unnormalized goals
        goals_sample_unnormalized = goals_sample.clone()

        # print(states_sample)
        # print(states_sample.shape)

        # normalize data
        states_sample = self.preproc_og(states_sample)
        goals_sample = self.preproc_og(goals_sample)
        s_prime_sample = self.preproc_og(s_prime_sample)
        # states_sample = self.o_norm.normalize(states_sample)
        # goals_sample = self.g_norm.normalize(goals_sample)
        # s_prime_sample = self.o_norm.normalize(s_prime_sample)

        # critic loss
        horizons_sample = horizons_sample.reshape((-1, 1))
        x_lhs = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                           torch.tensor(horizons_sample).float(), torch.tensor(actions_sample).float()),
                          dim=1).to(self.device)

        accessibilities_lhs1 = self.model.critic1(x_lhs)
        accessibilities_lhs2 = self.model.critic2(x_lhs)

        with torch.no_grad():
            x_rhs = torch.cat((torch.tensor(s_prime_sample).float(), torch.tensor(goals_sample).float(),
                               torch.tensor(horizons_sample - 1).float()),
                              dim=1).to(self.device)
            action_target = self.model.target_actor(x_rhs)#.detach().cpu()

            x_rhs2 = torch.cat((torch.tensor(s_prime_sample).float(), torch.tensor(goals_sample).float(),
                                torch.tensor(horizons_sample - 1).float(), action_target.float()),
                               dim=1).to(self.device)
            accessibilities_rhs1 = self.model.target_critic1(x_rhs2).detach()
            accessibilities_rhs2 = self.model.target_critic2(x_rhs2).detach()
            accessibilities_rhs = torch.min(accessibilities_rhs1, accessibilities_rhs2)

        # when h=1 or s_prim =g we are having a different target.
        for i in range(len(horizons_sample)):
            if self.vec_env.compute_reward(torch.unsqueeze(goal_achieved_sample[i], 0), torch.unsqueeze(goals_sample_unnormalized[i], 0)) > -0.5:
                accessibilities_rhs[i] = 1.0

            if horizons_sample[i] == 1:
                if self.vec_env.compute_reward(goal_achieved_sample[i], goals_sample_unnormalized[i]) > -0.5:
                    accessibilities_rhs[i] = 1.0
                else:
                    accessibilities_rhs[i] = 0.0

        self.optimizer_critic1.zero_grad()
        loss_this_batch1 = self.training_loss(accessibilities_lhs1, accessibilities_rhs)
        loss_this_batch1.backward()
        self.writer.add_scalar('losses/critic1_loss', loss_this_batch1)
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        loss_this_batch2 = self.training_loss(accessibilities_lhs2, accessibilities_rhs)
        loss_this_batch2.backward()
        self.writer.add_scalar('losses/critic1_loss', loss_this_batch1)
        self.optimizer_critic2.step()

        # actor_loss
        if self.train_step_count % self.policy_freq == 0:
            x_a = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                             torch.tensor(horizons_sample).float()),
                            dim=1).to(self.device)
            actions_actor = self.model.actor(x_a)#.cpu()

            x_a2 = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                              torch.tensor(horizons_sample).float(), actions_actor.float()),
                             dim=1).to(self.device)
            loss = -torch.mean(self.model.critic1(x_a2))
            self.writer.add_scalar('losses/actor_loss', loss)

            self.optimizer_actor.zero_grad()
            loss.backward()
            self.optimizer_actor.step()

        self.train_step_count += 1
    

    def train(self):
        # start_time_s = time()
        total_steps_taken = 0
        dataset = []
        self.init_tensors()
        def random_exploration_policy(states, goal):
            return np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=self.env_info["action_space"].shape)
        print("Random Exploration")
        for _ in range(self.random_exploration_eps):
            # print("Hello")
            ep_history, ep_success = self.run_episode(action_policy=random_exploration_policy,
                                            ep_length=self.max_ep_len)
            self.total_steps_taken += len(ep_history)

            # add episode to the dataset when the length is larger than 1
            if len(ep_history) > 1:
                dataset.extend(ep_history)

        # gde_start_time_s = time()
        print("Goal Directed Episodes")
        for gde_ep in range(self.goal_directed_eps):
            print(gde_ep)

            policy = self.goal_conditioned_c_learning_policy(rng=self.exploration_rng,
                                                        eval=False, exploration_epsilon=self.exploration_epsilon,
                                                        noise_epsilon=self.noise_epsilon, g_norm=self.g_norm, o_norm=self.o_norm)

            ep_history, ep_success = self.run_episode(action_policy=policy,
                                                ep_length=self.max_ep_len)

            # update statistics for o norm using the collected episode
            # self.update_o_norm(o_norm=self.o_norm, episode=ep_history)

            # add episode to the dataset when the length is larger than 1

            # TODO: Redundant. Is appending needed, or extending? Maybe I should keep a vector of stuff and unflatten it here
            if len(ep_history) > 1:
                dataset.extend(ep_history)

            # calc total steps taken
            self.total_steps_taken += len(ep_history) # TODO: make this 2D
            self.env_steps.append(self.total_steps_taken)

            # evaluation step
            if gde_ep % self.eval_freq == 0:

                eval_goal_step = []
                eval_goal_success = []
                goal_list = []
                for _ in range(self.num_eval_goals):
                    policy = self.goal_conditioned_c_learning_policy(rng=self.exploration_rng,
                                                                                        eval=True,
                                                                                        exploration_epsilon=self.exploration_epsilon,
                                                                                        noise_epsilon=self.noise_epsilon,
                                                                                        g_norm=self.g_norm, o_norm=self.o_norm)

                    ep_history, ep_success = self.run_episode(action_policy=policy,
                                                                            ep_length=self.max_ep_len)

                    eval_goal_step.append(len(ep_history))
                    eval_goal_success.append(ep_success) # TODO: extend this. pls

                # gde_elapsed_time_s = time() - gde_start_time_s
                # gde_mean_ep_time = gde_elapsed_time_s / (
                #             (gde_ep + 1) + (gde_ep // self.eval_freq + 1) * (1 + self.num_eval_goals))

                # print and save the result for the eval
                print(
                    f'Mean success rate to goal: {np.mean(eval_goal_success):.2f}\tMean step to goal: {np.mean(eval_goal_step):.1f} ' +
                    f' [episode {gde_ep} / {self.goal_directed_eps}, {0} s per episode]')

                self.eval_goal_list.append(np.array(goal_list))
                self.eval_mean_success.append(np.mean(eval_goal_success))
                self.eval_mean_step.append(np.mean(eval_goal_step))
                self.eval_pts.append(gde_ep)

            #train the models
            for _ in range(self.train_steps_per_ep):
                # sample a mini-batch of training
                batch_size = self.batch_size
                self.learn(gde_ep, dataset, batch_size=batch_size, batch_sampling_rng=self.batch_sampling_rng, o_norm=self.o_norm, g_norm=self.g_norm)

        # elapsed_time_s = time() - start_time_s
        # print('Training took {:.0f} seconds.'.format(elapsed_time_s))


        
