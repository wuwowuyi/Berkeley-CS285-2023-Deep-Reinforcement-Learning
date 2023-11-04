

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
![halfcheetah multi iter](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/cheetah_multi_iter.png)

## Problem 4
`python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_ablation.yaml`

#### Effect of ensemble size
Orange line has ensemble size is 3 (the baseline)
Blue line has ensemble size 1
Red line has ensemble size 10
We can see ensemble size 3 has the best performance, but no big difference from ensemble size 1 or 10.

![reacher ablation ensemble](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/reacher_ablation_ensemble.png)

#### Effect of planning horizon
Orange line's horizon is 10 (the baseline)
Blue line's horizon is 5, has the best performance.
Pink line's horizon is 20.

Horizon 5 has the best performance since in random shooting method, the further we plan into future, the more deviation we have.

![reacher ablation horizon](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/reacher_ablation_horizon.png)

#### Effect of the number of candidate action sequence

Orange line has 1000 candidate action sequences. (the baseline)
Green line has 2000 candidate action sequences.
Gray line has 500.
The performance of 500 candidate action sequences is significantly lower than the other two. 

![reacher ablation number of sequence](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw4/result_plots/reacher_ablation_num_seq.png)

