## My Note

## Experiments
**Experiments use hyperparameters provided by assignment, otherwise specified**.

### Experiment 1. Cartpole-v0
Average return vs. number of environment steps for CartPole_v0 small batch experiments. 
* Dark blue line, only normalize advantages
* Orange line, only reward-to-go
* Red line, both normalize advantages and reward-to-go
* Gray line, neith normalize advantages or reward-to-go

![CartPole small batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_small.png)

Average return vs. number of environment steps for CartPole_v0 large batch experiments.
* Green line, only normalize advantages (most stable performance)
* Pink line, only reward-to-go (also converge fast, but not as stable as green line.)
* Gray line, both reward-to-go and normalize advantages (converge relatively slower)
* Blue line, has neither, sort of fluctuating. 

![CartPole large batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_large.png)

#### CartPole Summary
In the CartPole env experiments, large batch experiments generally have better performance than small batch ones, especially when not use reward-to-go or normalization.

### Experiment 2. HalfCheetah-v4 
Below is the baseline loss:
![HalfCheetah baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/baseline_loss.png)

Below is the eval average return. Red line is the baseline version, orange line has no baseline.
![HalfCheetah average return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/eval_average_return.png)

Below are decreased baseline learning rate or gradient steps.
* the red line uses hyperparameters provided by assignment
* the green line decreases baseline learning rate from 0.01 to 0.005.
* the pink line decreases baseline gradient steps from 5 to 1.

We can see that decreasing baseline learning rate or gradient steps decreases performance.

![HalfCheetah decreased baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning_loss.png)
![HalfCheetah decreased baseline learning](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning.png)

Normalize advantages can improve performance, especially when there is no baseline, as shown below.
* light blue line has both baseline and normalized advantages.
* red line has only baseline
* dark blue line has no baseline, but normalized advantages.
* orange line has neither.

![HalfCheetah normalize advantage](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/normalize_advantage.png)


### Experiment 3. LunarLander-v2
hyperparameters provided by assignment: `--ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda <λ>`

Average return with different λ (seed=1):
* pink line, λ=1
* light blue line, λ=0.99
* red line, λ=0.98
* green line, λ=0
* dark blue line, λ=0.95

λ=1 and λ=0.99 have the best performance. λ=1 takes into account of all future rewards until terminal, λ=0.99 considers roughly the next 100 steps.

![LunarLander return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/lunar_lander_return.png)

λ=1 is equivalent to Monte Carlo advantage, i.e., `advantage(t) = Q(a_t,s_t) - V(s_t)`

λ=0 is equivalent to one step delta (TD error), i.e., `advantage(t) = reward(t) + gamma * V(s_t+1) - V(t)`

### Experiment 4. InvertedPendulum-v4

The best hyperparameter setting that reaches maximum performance with as few environment steps as possible is `--discount 0.9 --gae_lambda 0.99 -rtg --use_baseline -na -lr 0.01 -n 60 --batch_size 2500`. Below is average return with 5 different seeds.

![InvertedPendulum default return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/ip_best_return.png)

With default settings, i.e., `-rtg --use_baseline -na -n 100 --batch_size 5000`, below is the average return over 5 different seeds:
![InvertedPendulum default return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/ip_default_return.png)

Two best runs of best hyperparameters and default.

![InvertedPendulum default return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/ip_two_best.png)
