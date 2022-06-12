from rcrl.envs.naive_cs_env import NaiveCSEnv
from rcrl.rcsac_agent import RepConstraintSACAgent
from rcrl.sac_ae import ReplayBuffer
from rcrl.logger import Logger

import numpy as np
import torch
import argparse
import os
import os.path as osp
import gym
import time
import json
import random

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        # print("Make dir error:", dir_path)
        pass
    return dir_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='ncs, walker')
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir',  default='exp', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    # train
    parser.add_argument('--num_train_steps', default=100, type=int)
    parser.add_argument('--log_freq', default=20, type=int)
    # parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
    parser.add_argument('--init_steps', default=10, type=int)
    # parser.add_argument('--batch_size', default=512, type=int)
    # parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    # parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    # parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # encoder
    parser.add_argument('--encoder_feature_dim', default=4, type=int)
    args = parser.parse_args()
    return args
    

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

class EvalMode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )

def evaluate(env, agent):
    pass

def make_agent(obs_shape, action_shape, args, device):
    pass



def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer_capacity = int(1e6) 
    batch_size = 64

    if args.task == 'ncs':
        env = NaiveCSEnv()
        eval_env = NaiveCSEnv()
    elif args.task == 'walker':
        env_name="Walker2d-v2"
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    else:
        raise NotImplementedError

    replay_buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=replay_buffer_capacity,
        batch_size=batch_size,
        device=device
    )

    agent = RepConstraintSACAgent(
        obs_shape=env.observation_space.shape[0],
        action_shape=env.action_space.shape[0],
        log_freq=args.log_freq,
        device=device,
        encoder_feature_dim=args.encoder_feature_dim,
    )

    L = Logger(args.work_dir)
    model_dir = make_dir(osp.join(args.work_dir, 'model'))
    buffer_dir = make_dir(osp.join(args.work_dir, 'buffer'))

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with EvalMode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1





if __name__ == '__main__':
    main()
