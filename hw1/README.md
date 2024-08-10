## Behavioral Cloning

| env            | return mean vs expert mean | return std vs expert std | eval_batch_size |
|----------------|----------------------------|--------------------------|-----------------|
| ant-v4         | 4640  vs. 4682             | 118.60 vs. 30.71         | 5000            |
| halfcheetah-v4 | 3957  vs. 4035             | 65.75  vs. 32.87         | 5000            |
| hoppper-v4     | 1060  vs. 3718             | 75.49  vs. 0.35          | 2000            |
| walker2d-v4    | 244.7 vs. 5383             | 301.5  vs. 54.15         | 2000            |


## DAgger

10 iterations. 
eval_batch_size 5000.

### ant-v4 return mean and std

![ant mean return](result_plots/ant_average_return.png)

![ant std return](result_plots/ant_return_std.png)


### HalfCheetah return mean and std

![halfcheetah mean return](result_plots/halfcheetah_average_return.png)

![halfcheetah std return](result_plots/halfcheetah_return_std.png)

### Hopper-v4 return mean and std

![hopper mean return](result_plots/hopper_average_return.png)

![hopper std return](result_plots/hopper_return_std.png)

### Walker-4 return mean and std

![walker mean return](result_plots/walker_average_return.png)

![walker std return](result_plots/walker_return_std.png)

