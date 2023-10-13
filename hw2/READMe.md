## My Note

## Experiments

### 3.2 Cartpole Experiments
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

#### CartPole Summary
In the CartPole env experiments, Normalize advantages is more important than reward-to-go to reduce variance. When training on large batches, the model seems to "overfit". We either need to reduce learning rate or stop training.

### 4.2 HalfCheetah Experiments
![HalfCheetah baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/baseline_loss.png)

Above is the baseline loss, using hyperparameters provided by assignment, i.e., `--env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline`.

![HalfCheetah average return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/eval_average_return.png)

Above is the eval average return. Red line is the baseline verion, orange has no baseline. Both use hyperparameters provided by assignment.

![HalfCheetah decreased baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning_loss.png)
![HalfCheetah decreased baseline learning](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning.png)

Above are decrease baseline learning rate or steps.
* the red line uses hyperparameters provided by assignment
* the green line decreases baseline learning rate from 0.01 to 0.005.
* the pink line decreases baseline gradient steps from 5 to 1.
We can see to decrease baseline learning rate and gradient steps decrease performance.

![HalfCheetah normalize advantage](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/normalize_advantage.png)

Normalize advantages can improve performance, especially when there is no baseline.
* light blue line has both baseline and normalized advantages.
* red line has only baseline
* dark blue line has no baseline, but has normalized advantages.
* orange line has neither.
All are on hyperparameters provided by assignment.






