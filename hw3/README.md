
## Deep Q-learning

### Basic Q-learning
DQN algorithm implementation.

#### CartPole-v1 eval return over 3 seeds, 1, 2, and 3.
`python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml --seed <seed>`

![CartPole](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole.png)

#### LunarLander-v2 eval return over 3 seeds, 1, 2, and 3.
`python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed <seed>`

![LunarLander](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/lunarlander.png)

#### CartPole-v1 with learning rate increased to 0.05
eval return over 3 seeds, a lot lower than with learning rate 0.001.
![CartPole increased learning rate](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-lr.png)

Q-values comparison. blue line is learning rate 0.05, red line 0.001.

![CartPole q-values comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-qvalues.png)

Critic loss comparison, blue line is learning rate 0.05, red line 0.001.

![CartPole critic comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/cartpole-critic.png)

With learning rate 0.05, Q-values seem severely overestimated, so as critic loss.

### Double Q-learning
#### LunarLander-v2 double Q-learning 

Eval return over 3 different seeds.

![LunarLander comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/lunarlander-dq-eval.png)

Train return over 3 different seeds.

![LunarLander comparison](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/lunarlander-dq-train.png)

The reason for the difference, I understand, is because in training action selection is epsilon-greedy, and in eval action selection is deterministic.

#### MsPacman

![MsPacman train return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/mspacman_train_return.png)
![MsPacman eval return](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw3/result_plots/mspacman_eval_return.png)

During early training, the eval_return increases, but train_return is flattened. This is because when sample an action from the DQN agent, training uses a higher epsilon (close to 1) than the default value (0.02) used by eval. 

