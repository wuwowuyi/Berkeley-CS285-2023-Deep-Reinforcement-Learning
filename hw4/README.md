

# Model Based Reinforcement Learning Code

## Problem 1
`python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml`
All default hyperparameters
![halfcheetah_0_iter default](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/data/cheetah-cs285-v0_cheetah_0iter_l1_h32_mpcrandom_horizon10_actionseq1000_02-11-2023_16-48-34/itr_0_loss_curve.png)

num_layers: 2, hidden_size: 64
![halfcheetah_0_iter default](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/data/cheetah-cs285-v0_cheetah_0iter_l2_h64_mpcrandom_horizon10_actionseq1000_02-11-2023_20-22-07/itr_0_loss_curve.png)

num_layers: 2, hidden_size: 64, learning_rate: 0.0001, num_agent_train_steps_per_iter: 1000
![halfcheetah_0_iter default](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/data/cheetah-cs285-v0_cheetah_0iter_l2_h64_mpcrandom_horizon10_actionseq1000_02-11-2023_20-22-07/itr_0_loss_curve.png)

## Problem 2
`python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml`
Average eval return: -41.16.

## Problem 3

`python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_multi_iter.yaml`
![obstacles multi iter](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/obstacles_multi_iter.png)

`python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_multi_iter.yaml`
![reacher multi iter](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/reacher_multi_iter.png)

`python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml`
![reacher multi iter](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/cheetah_multi_iter.png)
