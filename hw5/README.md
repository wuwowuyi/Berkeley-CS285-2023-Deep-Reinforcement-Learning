
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
CQL trained on PointmassHard-v0, offline datasets 20K steps.
training_steps: 200000, hidden_size: 64, num_layers: 2
Varying cql_alpha values in [0, 0,1, 0.2, 0,5, 1.0, 2.0, 5.0, 10.0]
The light blue line and pink line have the best performance, with cql_alpha 0.5 and 1.0 respectively.
With cql_alpha=0, i.e., DQN is the orange line.


Evaluation trajectories of the pink line (cql_alpha=0.5)


### IQL and AWAC



## Problem 3: Online Fine-Tuning
### CQL
Environment PointmassHard-v0, Offline dataset 20k, offline_steps 100k, total_steps 200k
cql_alpha: 1.0, do not see performance difference with different epsilon schedule.


Corresponding Epsilon schedule


Evaluation trajectories examples


### AWAC
Environment PointmassHard-v0, Offline dataset 20k, offline_steps 100k, total_steps 200k




