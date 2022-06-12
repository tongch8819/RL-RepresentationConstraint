from rcrl.envs import NaiveCSEnv 
from rcrl.load_model import load_model

import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp

total_steps = 500


def get_action(obs, models):
    actor, constraint = models['actor'], models['constraint']
    # run on cpu
    constraint_rnds = 5
    with torch.no_grad():
        obs = torch.FloatTensor(obs)
        obs = obs.unsqueeze(0)
        for _ in range(constraint_rnds):
            mu, pi, _, _ = actor(obs, compute_log_pi=False)
            constraint_vec = constraint(pi)
            if (constraint_vec > 0.0).sum() == 0:
                break
        return pi.cpu().data.numpy().flatten()

def get_system_dr(state):
    return state.deploy_vec.sum() / state.physical_resource[0]

def run_one_round(seed, models : dict, verbose=False, save_action_hist=None, dump_data_path=None, sensitivity_tpl=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = NaiveCSEnv()
    assert 'actor' in models
    assert 'constraint' in models

    cache = dict()
    dr_hist = []
    a_hist = []
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    for i in range(total_steps):
        # Take deterministic actions at test time 
        a = get_action(o, models)
        o, r, d, info = env.step(a)
        dr = get_system_dr(env.cur_state)
        dr_hist.append(dr)
        a_hist.append(a)
        ep_ret += r
        ep_len += 1
        if verbose is not None:
            if verbose >= 1:
                print("Reward:", r, info['r_comps'])
            if verbose >= 2:
                print("Request: ", env.cur_state.request_vec)
                dep_vec = env.cur_state.deploy_vec
                dep_total = dep_vec.sum()
                print("Deploy: ", dep_vec, dep_vec.sum())
                allo_vec = env.cur_state.allocated_vec
                allo_total = allo_vec.sum()
                print("Allocated: ", allo_vec, allo_total)
                w = env.cur_state.physical_resource
                print("Ava, Phy, ActFactor: ", env.cur_state.available_quota, 
                    w,  allo_total / w )
                print("Allocation: ", a[:-1], "Factor: ", a[-1])
                print("Dep: ", dep_total / w)
                print()


    if verbose:
        print(f"Seed: {seed}   DR: {dr:.2%}")
    cache['dr_hist'] = dr_hist

    if save_action_hist is not None:
        pd.DataFrame(a_hist).to_csv(save_action_hist)
        print("Please check {} for action history.".format(save_action_hist))
    if dump_data_path:
        for name, user in env.user_pool.items():
            user.dump_hist_data(save_path=osp.join(dump_data_path, name + '.csv'))
        print("Please check {} for user trace.".format(dump_data_path))
    return cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str, help="")
    parser.add_argument('--dump_data_path', type=str, help="", default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='count', help='')
    parser.add_argument('--save_action_hist', type=str, help='Save action history path', default=None)
    parser.add_argument('--sensitivity_tpl', type=str, help='sensitivity tuple a_min,a_max', default=None)
    args = parser.parse_args()

    if args.sensitivity_tpl:
        sen_arg = list(map(float, args.sensitivity_tpl.split(',')))
    else:
        sen_arg = None
    models = load_model(args.work_dir)
    cache = run_one_round(seed=args.seed, models=models, 
        verbose=args.verbose, save_action_hist=args.save_action_hist, 
        dump_data_path=args.dump_data_path,
        sensitivity_tpl=sen_arg)
    print(len(cache['dr_hist']))

if __name__ == "__main__":
    main()