from constants import *
import numpy as np

## Describes the MDP data structure
class myMDP:
	"""Class for representing an MDP"""
	def __init__(self, numStates, numActions, rewards, transitionProbabilities, discountFactor, filename):
		global Rmax
		global Vmax
		self.numStates = numStates
		self.numActions = numActions
		self.rewards = map(float, rewards)
		Rmax = max(self.rewards)
		Rmin = min(self.rewards)
		self.rewards = np.reshape(self.rewards, (numStates, numActions, numStates))
		self.transitionProbabilities = map(float, transitionProbabilities)
		self.transitionProbabilities = np.reshape(self.transitionProbabilities, (numStates, numActions, numStates))
		self.discountFactor = discountFactor
		#### Normalising rewards to [0, 1]
		self.rewards = np.array(self.rewards)
		self.rewards = (self.rewards - float(Rmin))/(float(Rmax) - float(Rmin))
		Rmax = np.amax(self.rewards)
		Vmax = Rmax/(1-self.discountFactor)
		#print "Setting Vmax to be", Vmax
		self.Vmax = Vmax
		self.filename = filename
		#print "Initialised MDP with ", numStates, " states and ", numActions, " actions"
		#print self.rewards, "rewards"

	def printMDP(self):
		print "The given MDP has : ", self.numStates, " states"
		print self.numActions, " actions"
		print self.rewards, " rewards"
		print self.transitionProbabilities, " Probabilities"
		print self.discountFactor, " discount factor"
		print "Read from file ", self.filename

	def simulate(self, state, action):
		if (state>=self.numStates or state<0):
			print "The given state number does not exist"
			return None

		elif (action>=self.numActions or action<0):
			print "The given action does not exist"
			return None
		
		else:
			startIndex = state*self.numStates*self.numActions + action*self.numStates
			# probs = self.transitionProbabilities[startIndex:startIndex+self.numStates]
			probs = self.transitionProbabilities[state][action]
			# rewards = self.rewards[startIndex:startIndex+self.numStates]
			rewards = self.rewards[state][action]
			sampledstate = np.random.choice(np.arange(self.numStates), p=probs)
			return sampledstate, rewards[sampledstate]

	def getNumStates(self):
		return self.numStates

	def getNumActions(self):
		return self.numActions

def UniformSampling(times, mdp):
	"""Adopt a uniform sampling strategy and return the policy"""
	policy_answer = []
	for s in range(mdp.numStates):
		avg_rewards = [0 for i in range(mdp.numActions)]
		for a in range(mdp.numActions):
			for i in range(times):
				newstate, reward = mdp.simulate(s, a)
				avg_rewards[a] += int(reward)
			avg_rewards[a] /= times
		policy_answer.append(avg_rewards.index(max(avg_rewards)))
	return policy_answer