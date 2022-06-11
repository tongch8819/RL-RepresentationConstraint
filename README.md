# Representation Learning With Constraints

```
| train | E: 32 | S: 582 | D: 0.0 s | R: 2.5197 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 33 | S: 604 | D: 0.0 s | R: -1.6787 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 34 | S: 623 | D: 0.0 s | R: 6.6413 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 35 | S: 641 | D: 0.0 s | R: 3.6380 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 36 | S: 660 | D: 0.0 s | R: -2.8483 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 37 | S: 669 | D: 0.0 s | R: 4.8992 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 38 | S: 691 | D: 0.0 s | R: -5.2275 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 39 | S: 703 | D: 0.0 s | R: -6.5630 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 40 | S: 717 | D: 0.0 s | R: 1.6443 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 41 | S: 732 | D: 0.0 s | R: 1.4232 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| eval | S: 732 | ER: 0.0000
```


# TODO List

+ standalone encoder
+ bisimulation metric, wasserstein distance
+ contraint network

```
Traceback (most recent call last):
  File "train.py", line 204, in <module>
    main()
  File "train.py", line 182, in main
    agent.update(replay_buffer, L, step)
  File "/home/v-tongcheng/Projects/Rep_RL/rcrl/rcsac_agent.py", line 233, in update
    self.update_critic(obs, action, reward, next_obs, not_done, L, step)
  File "/home/v-tongcheng/Projects/Rep_RL/rcrl/rcsac_agent.py", line 154, in update_critic
    self.critic.log(L, step)
  File "/home/v-tongcheng/Projects/Rep_RL/rcrl/sac_ae.py", line 174, in log
    self.encoder.log(L, step, log_freq)
  File "/anaconda/envs/rl/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1186, in __getattr__
    type(self).__name__, name))
AttributeError: 'Encoder' object has no attribute 'log'
```