
## Deep Q-learning

### Basic Q-learning
DQN algorithm implementation.

#### CartPole-v1 eval return over 3 seeds, 1, 2, and 3.
`-cfg experiments/dqn/cartpole.yaml`

![CartPole](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole.png)

#### LunarLander-v2 eval return over 3 seeds, 1, 2, and 3.
`-cfg experiments/dqn/lunarlander.yaml`

![LunarLander](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/lunarlander.png)

#### CartPole-v1 with learning rate increased to 0.05
eval return over 3 seeds, a lot lower than with learning rate 0.001.
![CartPole increased learning rate](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-lr.png)

Q-values comparison. blue line is learning rate 0.001, red line 0.05.

![CartPole q-values comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-qvalues.png)

Critic loss comparison, blue line is learning rate 0.001, red line 0.05.

![CartPole critic comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-critic.png)

With learning rate 0.05, Q-values seem severely overestimated, so as critic loss.

