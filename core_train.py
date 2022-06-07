from rcrl.rcsac_agent import RepConstraintSACAgent
from rcrl.sac_ae import ReplayBuffer
from rcrl.logger import Logger

import numpy as np
import torch
import argparse
import os
import gym
import time
import json
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

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
    if args.agent == 'rcsac':
        agent = RepConstraintSACAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
        )
        
    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_dim = 4
    env_name="Walker2d-v2"
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger()
    model_dir = ''
    buffer_dir = ''

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
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
