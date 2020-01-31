from constants import *
import sys
import random
from MarkovChain import markovchain
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
	eps = eps_values[mdpname]
	start_state = 0

	UncContri = np.zeros((len(seeds), MAX_ITERATION_LIMIT + 1))
	ImpUnc = np.zeros((len(seeds), MAX_ITERATION_LIMIT + 1))
	uni = np.zeros((len(seeds), MAX_ITERATION_LIMIT + 1))
	epi = np.zeros((len(seeds), MAX_ITERATION_LIMIT + 1))

	for i in range(len(seeds)):
		randomseed = seeds[i]
		UncContri[i] = markovchain(mdp, start_state, eps, randomseed, "unc_contri")
		ImpUnc[i] = markovchain(mdp, start_state, eps, randomseed, "runcertainty")
		uni[i] = markovchain(mdp, start_state, eps, randomseed, "uniform")
		epi[i] = markovchain(mdp, start_state, eps, randomseed, "episodic")


	pickle_save(UncContri, f"results/markovchain/{mdpname}_UncContri_{len(seeds)}seeds.pkl")
	pickle_save(ImpUnc, f"results/markovchain/{mdpname}_ImpUnc_{len(seeds)}seeds.pkl")
	pickle_save(uni, f"results/markovchain/{mdpname}_uni_{len(seeds)}seeds.pkl")
	pickle_save(epi, f"results/markovchain/{mdpname}_epi_{len(seeds)}seeds.pkl")

	UncContri = np.abs(UncContri)
	ImpUnc = np.abs(ImpUnc)
	uni = np.abs(uni)
	eps = np.abs(eps)

	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], 100 * np.mean(ImpUnc, axis = 0)[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='ImpUnc')
	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], 100 * np.mean(uni, axis = 0)[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='Unifrom')
	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], 100 * np.mean(epi, axis = 0)[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='Episodic')
	plt.plot(1 + np.arange(MAX_ITERATION_LIMIT//2)[mdp.numStates * mdp.numActions + 500:], 100 * np.mean(UncContri, axis = 0)[mdp.numStates * mdp.numActions + 500: MAX_ITERATION_LIMIT//2], label ='UncContri')

	plt.plot(np.zeros(MAX_ITERATION_LIMIT//2))
	plt.xlabel('samples')
	plt.ylabel('Average Absolute Error % in Value Function')
	plt.legend(loc = 'upper left')
	plt.savefig(f"results/markovchain/graphs/{mdpname}_500_500000_avg_{len(seeds)}seeds.png")



if __name__ == '__main__':
	if(len(sys.argv)<2):
		print("Usage : python main.py <mdpfile>")
	else:
		main(sys.argv)