from constants import *
import sys
import random
from MDPclass import myMDP
from FIECHTER import FeichterPolicy
from RR import RoundRobin
from LUCB import LUCBStopping
from LUCBepisodic import LUCBEpisodic
from LUCBEpisodicBound import LUCBBound 
from MBIE import mbie
from DDVOuu import ddvouu
from PolicyIt import policyIt
from MarkovChainEsti import markovchainesti
import numpy as np

def main(argv):
	# print "Executing MDP"
	print(argv[1][argv[1].find('/')+1:])
	mdpname = argv[1][argv[1].find('/')+1:]
	lines = [line.rstrip('\n') for line in open(argv[1])]
	print(argv[2])

	global Rmax
	global Vmax
	numStates = int(lines[0])
	numActions = int(lines[1])
	rewards = np.array(lines[2].split())
	print(rewards)
	print(min(rewards))

	# rewards = np.reshape(rewards, (numStates,numActions,numStates))
	transitionProbabilities = np.array(lines[3].split())
	# transitionProbabilities = np.reshape(transitionProbabilities, (numStates,numActions,numStates))
	discountFactor = float(lines[4])
	filename = mdpname[mdpname.find('-')+1:mdpname.find('.')]
	theMDP = myMDP(numStates, numActions, rewards, transitionProbabilities, discountFactor, filename)
	theMDP.printMDP()
	eps = 0.1
	# if(mdpname=="mdp-riverswim.txt"):
	# 	start_state = random.randint(0,1)
	# else:
	# 	start_state = 0
	start_state = 0

	print(seeds)
	for randomseed in seeds:
		if(argv[2]=="uniform"):
			print("Doing naive uniform sampling")
			print("Final policy is : ", UniformSampling(100, theMDP))
		elif(argv[2]=="fiechter"):
			print("Doing Fiechter algorithm")
			print("Final policy is : ", FeichterPolicy(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="rr"):
			print("Doing Round robin")
			print("Final policy is : ", RoundRobin(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="lucb"):
			print("Doing LUCB type algorithm")
			print("Final policy is : ", LUCBStopping(theMDP, start_state, eps))
		elif(argv[2]=="lucb-eps"):
			print("Doing LUCB episodic type algorithm")
			print("Final policy is : ", LUCBEpisodic(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="lucb-bou"):
			print("Doing LUCB epi with MBIE bound type algorithm")
			print("Final policy is : ", LUCBBound(theMDP, start_state, eps))
		elif(argv[2]=="mbie"):
			print("Doing MBIE-reset algorithm")
			print("Final policy is : ", mbie(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="ddv-ouu"):
			print("Doing DDV-OUU algorithm")
			print("Final policy is : ", ddvouu(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="policy"):
			print("Doing policy iteration")
			print("Final policy is : ", policyIt(theMDP, start_state, eps, randomseed))
		elif(argv[2]=="markov_esti"):
			print("Doing markov chain estimation")
			if str(argv[3]) in ["use_ddv","episodic","uniform","greedyMBAE","greedyMBIE","mybest","runcertainty", "unc_contri", "rupper_uncertainty"]:
				print("Final policy is : ", markovchainesti(theMDP, start_state, eps, randomseed, str(argv[3])))
			else:
				print("Please choose a recognized markov algorithm from [use_ddv,episodic,uniform,greedyMBAE,greedyMBIE,mybest]")
		else:
			print("Unrecognized algorithm!")
			print("Please try one of [uniform, fiechter, rr, lucb, mbie, ddv-ouu, policy]")

	


def UniformSampling(times, mdp):
	"""Adopt a uniform sampling strategy and return the policy"""
	policy_answer = []
	for s in range(mdp.numStates):
		avg_rewards = [0 for i in range(mdp.numActions)]
		for a in range(mdp.numActions):
			for i in range(times):
				newstate, reward = mdp.simulate(s, a)
				avg_rewards[a] += int(reward)
				#print(s, a, reward)
				print(avg_rewards)
			avg_rewards[a] = float(avg_rewards[a])/times
		print(avg_rewards)
		policy_answer.append(avg_rewards.index(max(avg_rewards)))
	return policy_answer

if __name__ == '__main__':
	if(len(sys.argv)<3):
		print("Usage : python main.py <mdpfile> <algorithm> [uniform, fiechter, rr, lucb, mbie, ddv-ouu, policy] <epsilon>")
	else:
		main(sys.argv)