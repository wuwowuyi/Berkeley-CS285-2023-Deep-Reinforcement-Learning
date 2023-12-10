
# Exploration Strategies and Offline Reinforcement Learning
Code results

## Problem 1: Exploration

PointmassEasy-v0, Random network distillation heatmap, 10k total steps
![easy heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_easy.png)

PointmassMedium-v0, Random network distillation heatmap, 10k total steps
![medium 10k step heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_medium_10k.png)

PointmassMedium-v0, Random network distillation heatmap, 20k total steps
![medium 20k step heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_medium_20k.png)

PointmassHard-v0, Random network distillation heatmap, 10k total steps
![hard 10k step heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_hard_10k.png)

PointmassHard-v0, Random network distillation heatmap, 20k total steps
![hard 20k step heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_hard_20k.png)

PointmassHard-v0, Random network distillation heatmap, 50k total steps
![hard 50k step heatmap](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/exploration_hard_50k.png)


## Problem 2: Offline RL
### CQL
CQL trained on PointmassHard-v0, offline datasets 10K steps.
training_steps: 200000, hidden_size: 64, num_layers: 2
Varying cql_alpha values in [0, 0,1, 0.2, 0,5, 1.0, 2.0, 5.0, 10.0]
The red line and light blue line have the best performance, with cql_alpha 0.5 and 1.0 respectively.
With cql_alpha=0, i.e., DQN is the orange line.
![cql 10k step, multiple alpha](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/offline_cql_hard_10k_alpha_eval.png)

Evaluation trajectories of the red line (cql_alpha=0.5)
![cql 10k step eval trajectories](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/offline_cql_hard_10k_traj.png)

### IQL and AWAC
AWAC trained on PointmassHard-v0, with offline datasets 10K, 20k, and 50K steps respectively. The bigger the dataset size, the better performance the method has.
![awac hard different datasets](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/offline_awac_hard_eval.png)

Evaluation trajectories with 50k step dataset.
![awac hard 50k trajectories](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/offline_awac_hard_50k_traj.png)

## Problem 3: Online Fine-Tuning
### CQL
Environment PointmassHard-v0, Offline dataset 20k, offlin training steps 100k, total training steps 200k
cql_alpha is 1.0, do not see performance difference with different epsilon schedule.
![finetune cql hard 20k](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/finetune_cql_20k_hard_eval.png)

Corresponding Epsilon schedule
![finetune cql hard 20k epsilon](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/finetune_cql_epsilon.png)

State map from online training, 20k step offline dataset
![finetune cql online state map](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/finetune_cql_hard_20k_online_state_map.png)

### AWAC
Environment PointmassHard-v0, with offline datasets 10K, 20k, and 50K respectively, offline training steps 100k, total training steps 200k

![finetune awac](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/finetune_awac_hard_eval.png)

State map from online training, 50k step offline dataset
![finetune awac online training state map](https://github.com/wuwowuyi/Berkeley-CS285-Deep-Reinforcement-Learning/blob/learning/hw5/result_plots/finetune_awac_hard_50k_online_state_map.png)


