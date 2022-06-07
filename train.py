from rcrl.rcsac_agent import RepConstraintSACAgent
from rcrl.encoder import Encoder
import gym
import torch
import argparse

feature_dim = 4
env_name="Walker2d-v2"

def test():
    env = gym.make(env_name)
    print(env.observation_space)
    
    # obs = env.observation_space.sample()
    obs = env.reset()
    
    enc = Encoder(env.observation_space.shape[0], feature_dim, [64,64])
    print(enc.fc)
    
    x = torch.tensor(obs, dtype=torch.float32) # make sure the input datatype is float32 instead of double
    print(x.shape)
    zeta = enc(x)
    print(obs, zeta)

def train():
    env = gym.make(env_name)
    agent = RepConstraintSACAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    )
    agent.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='')
    args = parser.parse_args()

    if args.test:
        test()
        return

    train()
    
if __name__ == "__main__":
    main()