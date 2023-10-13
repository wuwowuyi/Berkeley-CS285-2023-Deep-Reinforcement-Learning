## My Note

## Experiments
**Experiments use hyperparameters provided by assignment, otherwise specified**.

### 3.2 Cartpole Experiments
Average return vs. number of environment steps for CartPole_v0 small batch experiments. 
![CartPole small batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_small.png)

* Dark blue line, only normalize advantages
* Orange line, only reward-to-go
* Red line, both normalize advantages and reward-to-go
* Gray line, neith normalize advantages or reward-to-go

Average return vs. number of environment steps for CartPole_v0 large batch experiments.
![CartPole large batch](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/cartpole_large.png)

* Green line, only normalize advantages (most stable performance)
* Pink line, only reward-to-go (also converge fast, but not as stable as green line.)
* Gray line, both reward-to-go and normalize advantages (converge relatively slower)
* Blue line, has neither, sort of fluctuating. 

#### CartPole Summary
In the CartPole env experiments, large batch generally has better performance than small batch, especially when not use reward-to-go or normalization.

### 4.2 HalfCheetah Experiments
Below is the baseline loss:
![HalfCheetah baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/baseline_loss.png)

Below is the eval average return. Red line is the baseline version, orange line has no baseline.
![HalfCheetah average return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/eval_average_return.png)

Below are decreased baseline learning rate or gradient steps.
* the red line uses hyperparameters provided by assignment
* the green line decreases baseline learning rate from 0.01 to 0.005.
* the pink line decreases baseline gradient steps from 5 to 1.

![HalfCheetah decreased baseline loss](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning_loss.png)
![HalfCheetah decreased baseline learning](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/decreased_learning.png)

We can see that decreasing baseline learning rate or gradient steps decreases performance.

Normalize advantages can improve performance, especially when there is no baseline.
* light blue line has both baseline and normalized advantages.
* red line has only baseline
* dark blue line has no baseline, but has normalized advantages.
* orange line has neither.

![HalfCheetah normalize advantage](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw2/normalize_advantage.png)






