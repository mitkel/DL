# Conditional variational autoencoders for stochastic processes

## Aim:
- Generate multiple trajectories of Ornstein-Uhlenbeck process
- Each trajectory has random starting point and random parameters
- Implement CVAE to simulate last T steps of a trajectory
- Apply CVAE decoder to generate future trajectories of test sample
- Compare CVAE and MCMC on pricing European and Asian options on test sample

## Starter (toy-model):
- Train and test sample trajectories have the same process parameters
