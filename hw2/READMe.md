## My Note

## Experiments

### Cartpole
Average return vs. number of environment steps for CartPole_v0 small batch experiments. 
![CartPole small batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_small.png)

All are on default parameters. From top to bottom are:
* Reward-to-go and normalize advantages are both true
* normalize advantages is true
* reward-to-go is true
* Reward-to-go and normalize advantages are both false

Average return vs. number of environment steps for CartPole_v0 large batch experiments.
![CartPole large batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_large.png)

All are on default parameters otherwise mentioned. From top to bottom:
* Green line, reward-to-go and normalize advantages are both true. (learning rate 0.003)
* Blue line, reward-to-go and normalize advantages are both true
* Orange line, normalize advantages is true
* Grey line, only reward_to_go
* bottom green line, reward-to-go and normalize advantages are both false

#### Summary
Normalize advantages is more important than reward-to-go to reduce variance.
When training on large batches, the model seems to "overfit". We either need to reduce learning rate or stop training.
