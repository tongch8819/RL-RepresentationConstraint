from rcrl.constraint import ConstraintNetwork
from rcrl.sac_ae_v2 import Actor, Critic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepConstraintSACAgent(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        log_freq,  # actor, critic, encoder, constraint log frequency
                   # different between train and test phase
        device='cpu',
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        constraint_num=5,
        constraint_lr=1e-3,
        constraint_rnds=10,
    ):
        self.log_freq = log_freq
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.bisim_coef = bisim_coef
        self.constraint_rnds = constraint_rnds

        self.actor = Actor(
            obs_shape, action_shape, encoder_feature_dim, 
            hidden_dim, num_layers,
            actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, encoder_feature_dim, 
            hidden_dim, num_layers
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, encoder_feature_dim, 
            hidden_dim, num_layers
        ).to(device)

        self.constraint_network = ConstraintNetwork(
            act_dim=action_shape, constraint_num=constraint_num
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.reward_decoder = nn.Sequential(
        #     nn.Linear(encoder_feature_dim, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)).to(device)

        # TODO: double or single?
        # tie encoders between actor and critic
        # self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.constraint_network_optimizer = torch.optim.Adam(
            self.constraint_network.parameters(), lr=constraint_lr,
        )

        # set modules in training mode
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        # function overloading
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # def select_action(self, obs):
        # with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            # obs = obs.unsqueeze(0)
            # mu, _, _, _ = self.actor(
                # obs, compute_pi=False, compute_log_pi=False
            # )
            # mu_star = self.constraint_network(mu)
            # return mu_star.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            for _ in range(self.constraint_rnds):
                mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
                constraint_vec = self.constraint_network(pi)
                if (constraint_vec > 0.0).sum() == 0:
                    break
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step, self.log_freq)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step, self.log_freq)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_constraint_network(self, obs, action, reward, L, step):
        # label the action
        constraint_vecs = self.constraint_network(action)
        # preds = (constraint_vecs <= 0.0).all(dim=1).to(dtype=torch.float32).unsqueeze(dim=1)
        preds = constraint_vecs.mean(dim=1, keepdim=True)
        labels =  ((reward - reward.mean()) > 0).to(dtype=torch.float32)
        loss = F.mse_loss(preds, labels)
        self.constraint_network_optimizer.zero_grad()
        loss.backward()
        self.constraint_network_optimizer.step()

        self.constraint_network.log(L, step, self.log_freq)

    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_constraint_network(obs, action, reward, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            self.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.constraint_network.state_dict(),
            '%s/constraint_network_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        # self.reward_decoder.load_state_dict(
            # torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        # )

    def soft_update_params(self, net, target_net, tau):
        """Update target network by polyak."""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


def main():
    agent = RepConstraintSACAgent(obs_shape=17, action_shape=4)
    print(agent)

if __name__ == "__main__":
    main()