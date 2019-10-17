import numpy as np
from constants import MAX_ITERATION_LIMIT, epsilon_convergence


def evaluatePolicy(mdp, policy, start_state):

	Vpred = np.zeros((mdp.numStates))
	Rsa = [[np.sum(mdp.rewards[i][j]*mdp.transitionProbabilities[i][j]) for j in range(mdp.numActions)] for i in range(mdp.numStates)]
	for i in range(MAX_ITERATION_LIMIT):

		oldVstar = np.copy(Vpred)
		for st in range(mdp.numStates):
			Vpred[st] = Rsa[st][policy[st]] + mdp.discountFactor*np.sum(mdp.transitionProbabilities[st][policy[st]]*Vpred)

		if(np.linalg.norm(oldVstar-Vpred)<=epsilon_convergence):
			break

	return Vpred[start_state]

