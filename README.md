# Representation Learning With Constraints

The log definition is in a format variable in `logger.py`.
```
| train | E: 32 | S: 582 | D: 0.0 s | R: 2.5197 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
| train | E: 33 | S: 604 | D: 0.0 s | R: -1.6787 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000 | MR: 0.0000
```


# TODO List

+ walker baseline training
+ deep mdp baseline 
+ standalone encoder
+ bisimulation metric, wasserstein distance


1. Since naive cloud env never done, train script cannot save the current model.
Fix it by add current step in CSEnv.
2. When episode is done the logs for train and evaluation would occur. 
Fix it by change debug of ncs into 1000 steps for at least one episode with done.
3. In walker task log frequency is too long to show enough information in tensorboard
Try to fix it by change log frequency in training of walker from 1000 into 100.
