from rcrl.sac_ae import Actor
from rcrl.constraint import ConstraintNetwork
from rcrl.envs.naive_cs_env import NaiveCSEnv
import torch
import os.path as osp

def load_model(work_dir):
    # arbitrary re-construction network should be consistent with train.py
    encoder_feature_dim = 4
    
    num_layers=4
    hidden_dim=256
    actor_log_std_min=-10
    actor_log_std_max=2
    env = NaiveCSEnv()
    actor = Actor(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        encoder_feature_dim, 
        hidden_dim, num_layers,
        actor_log_std_min, actor_log_std_max
    )
    actor.load_state_dict(torch.load(osp.join(work_dir, 'model/actor_0.pt')))
    actor.eval()  # make sure some layers like dropout and batch normalization are in evaluation mode

    constraint_network = ConstraintNetwork(
        act_dim=env.action_space.shape[0],
        constraint_num=5,
    )
    constraint_network.load_state_dict(torch.load(osp.join(work_dir, 'model/constraint_network_0.pt')))
    constraint_network.eval()

    return dict(actor=actor, constraint=constraint_network)

def main():
    out = load_model('exp/ncs')
    print(out)

if __name__ == "__main__":
    main()