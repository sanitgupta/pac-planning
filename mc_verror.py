from constants import *
import sys
import random
from MarkovChainEsti import markovchainesti
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from MDPclass import myMDP
from util import *


def main(argv):

	print(argv[1][argv[1].find('/')+1:])
	mdpname = argv[1][argv[1].find('/')+1:]
	lines = [line.rstrip('\n') for line in open(argv[1])]
	#print argv[2]

	global Rmax
	global Vmax
	numStates = int(lines[0])
	numActions = int(lines[1])
	rewards = np.array(lines[2].split())
	# rewards = np.reshape(rewards, (numStates,numActions,numStates))
	transitionProbabilities = np.array(lines[3].split())
	# transitionProbabilities = np.reshape(transitionProbabilities, (numStates,numActions,numStates))
	discountFactor = float(lines[4])
	filename = mdpname[mdpname.find('-')+1:mdpname.find('.')]
	mdp = myMDP(numStates, numActions, rewards, transitionProbabilities, discountFactor, filename)
	mdp.printMDP()
	#eps = eps_values[mdpname]
	eps = 0.1
	start_state = 0

	UCB_avg = np.zeros(MAX_ITERATION_LIMIT+1)
	uni_avg = np.zeros(MAX_ITERATION_LIMIT+1)
	epi_avg = np.zeros(MAX_ITERATION_LIMIT+1)

	for randomseed in seeds:
		#UncContri = markovchainesti(mdp, start_state, eps, randomseed, "unc_contri")
		#ImpUnc = markovchainesti(mdp, start_state, eps, randomseed, "runcertainty")
		UCB = markovchainesti(mdp, start_state, eps, randomseed, "rupper_uncertainty")
		uni = markovchainesti(mdp, start_state, eps, randomseed, "uniform")
		epi = markovchainesti(mdp, start_state, eps, randomseed, "episodic")

		UCB_avg += np.absolute(UCB)
		uni_avg += np.absolute(uni)
		epi_avg += np.absolute(epi)

		# #plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], ImpUnc[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='ImpUnc')
		# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], uni[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='Unifrom')
		# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], epi[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='Episodic')
		# #plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], UncContri[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='UncContri')
		# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], UCB[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='UCB')
		
		#plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], ImpUnc[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='ImpUnc')
		plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], uni[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='Unifrom')
		plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], epi[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='Episodic')
		#plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], UncContri[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='UncContri')
		plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], UCB[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='UCB')

		plt.plot(np.zeros(MAX_ITERATION_LIMIT))
		plt.xlabel('samples')
		plt.ylabel('Error % in value function')
		plt.legend(loc = 'upper left')
		plt.savefig(f"results/markovchain/graphs/{mdpname}_{MAX_ITERATION_LIMIT}_ucb_initsamples_randomseed_{randomseed}.png")
		plt.clf()

	UCB_avg = UCB_avg / len(seeds)
	uni_avg = uni_avg / len(seeds)
	epi_avg = epi_avg / len(seeds)
	
	# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], uni_avg[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='Unifrom')
	# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], epi_avg[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='Episodic')
	# #plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], UncContri[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='UncContri')
	# plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions + 500:], UCB_avg[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT], label ='UCB')

	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], uni_avg[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='Unifrom')
	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], epi_avg[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='Episodic')
	#plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], UncContri[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='UncContri')
	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT)[mdp.numStates * mdp.numActions+20:], UCB_avg[mdp.numStates * mdp.numActions+20:MAX_ITERATION_LIMIT], label ='UCB')

	plt.plot(np.zeros(MAX_ITERATION_LIMIT))
	plt.xlabel('samples')
	plt.ylabel('Error % in value function')
	plt.legend(loc = 'upper left')
	plt.savefig(f"results/markovchain/graphs/{mdpname}_{MAX_ITERATION_LIMIT}_ucb_initsamples_seeds_{len(seeds)}.png")
	plt.clf()

if __name__ == '__main__':
	if(len(sys.argv)<2):
		print("Usage : python mc_verror.py <mdpfile>")
	else:
		main(sys.argv)