INPUT FORMAT :::

Each MDP is provided as a text file in the following format.

Number of states
Number of actions
Reward function
Transition function
Discount factor
The number of states S and the number of actions A will be integers. Assume that the states are numbered 0,1,…,S−1, and the actions are numbered 0,1,…,A−1. The reward function will be provided as SAS entries. Each entry corresponds to R(s,a,s′), wherein state s, action a, and state s′ are being iterated in sequence from 0 to S−1, 0 to A−1, and 0 to S−1, respectively. A similar scheme is adopted for the transition function T. Rewards can be positive, negative, or zero. The discount factor is a real number between 0 (included) and 1 (excluded).