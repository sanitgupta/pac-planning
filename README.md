## Introduction ##

Markov Decision Processes (MDPs) are a fundamental mathematical abstraction used to model sequential decision making under uncertainty and are a
discrete model of discrete-time stochastic control and reinforcement learning problems. Particularly important in MDPs is the planning problem, wherein we try to compute an
optimal policy that maps each state of an MDP to an action to be followed
at that state. The goal in reinforcement learning is usually to find the optimal policy, i.e. a policy which maximises the
reward of traversing the MDP. 

We study the planning problem assuming that a near perfect simulator is available. Given how expensive it can be to gather data, the time required to find a near optimal policy for many problems is dominated by the number of calls to the simulator.  A good MDP planning
algorithm in such a setting attempts to minimise the number of calls made to the
simulator to learn a policy that is very close to being optimal with a
high probability. This is known as the probably approximately correctly (PAC) framework.


