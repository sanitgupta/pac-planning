import numpy as np

# Vmax = 10000
# Rmax = 100
MAX_ITERATION_LIMIT = 100000
c = 10000
PRINT_ITERATION = 10000
converge_iterations = 1000
epsilon_convergence = 1e-3
eps_values = {'mdp-01.txt':5*0.51794, 'mdp-02.txt':1.9667, 'mdp-03.txt':5*0.860558,'mdp-03-trunc.txt':5*0.860558, 'mdp-04.txt':1.47829, 'mdp-06.txt':0.1, 'mdp-riverswim.txt':5*0.153096, 'mdp-riverf.txt':5*0.153096, 'mdp-CasinoLand.txt':5*1.235594, 'mdp-SixArms.txt':5*0.82568}
#seeds = [10,20,30,40,50,60,70,80,90,100]
seeds = [20]
#seeds = np.arange(0, 10, 5)